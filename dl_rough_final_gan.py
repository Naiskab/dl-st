# ================================================================
# COLORIZATION OF GRAYSCALE IMAGES USING DEEP NEURAL NETWORKS
# ================================================================

import os
import numpy as np
from PIL import Image
from skimage import color
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
# Resize images to (256x256)
IMAGE_SIZE = 256
# EPOCHS = 1
EPOCHS = 50  # More epochs for GAN training
LR_G = 0.0002  # Generator learning rate
LR_D = 0.0002  # Discriminator learning rate (higher to help it keep up)
LR = 0.0001
BATCH_SIZE = 16
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

LAMBDA_L1 = 1  # Weight for L1 loss
D_STEPS = 1  # Discriminator steps per generator step
LABEL_SMOOTHING = 0.1  # One-sided label smoothing for real images

import csv
import matplotlib.pyplot as plt

# Create CSV file for logging
def init_loss_log(filename='training_losses.csv'):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'G_Loss', 'D_Loss', 'L1_Loss', 'GAN_Loss'])

def log_losses(epoch, g_loss, d_loss, l1_loss, gan_loss, filename='training_losses.csv'):
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, g_loss, d_loss, l1_loss, gan_loss])

# ------------------------------------------------
# DEFINE UTIL FUNCTIONS
# ------------------------------------------------

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:18])

    def forward(self, img):
        return self.feature_extractor(img)


# The function to center and normalise the images
class BaseColor(nn.Module):
    def __init__(self):
        super(BaseColor, self).__init__()
        self.l_cent = 50.
        self.l_norm = 100.
        self.ab_norm = 110.

    def normalize_l(self, in_l):
        return (in_l - self.l_cent) / self.l_norm

    def unnormalize_l(self, in_l):
        return in_l * self.l_norm + self.l_cent

    def normalize_ab(self, in_ab):
        return in_ab / self.ab_norm

    def unnormalize_ab(self, in_ab):
        return in_ab * self.ab_norm


normalizer = BaseColor()


def load_img(img_path):
    """
    Load an image from disk as a numpy RGB array.
    If grayscale, convert to 3-channel RGB by tiling.
    """
    out_np = np.asarray(Image.open(img_path))
    if out_np.ndim == 2:
        # if grayscale → replicate channel 3 times
        out_np = np.tile(out_np[:, :, None], 3)
    return out_np


class ECCVGenerator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(ECCVGenerator, self).__init__()

        # Model 1: 1 -> 64, downsample to 128x128
        model1 = [nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True), ]
        model1 += [nn.ReLU(True), ]
        model1 += [nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True), ]
        model1 += [nn.ReLU(True), ]
        model1 += [norm_layer(64), ]

        # Model 2: 64 -> 128, downsample to 64x64
        model2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True), ]
        model2 += [nn.ReLU(True), ]
        model2 += [nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True), ]
        model2 += [nn.ReLU(True), ]
        model2 += [norm_layer(128), ]

        # Model 3: 128 -> 256, downsample to 32x32
        model3 = [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True), ]
        model3 += [nn.ReLU(True), ]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), ]
        model3 += [nn.ReLU(True), ]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True), ]
        model3 += [nn.ReLU(True), ]
        model3 += [norm_layer(256), ]

        # Model 4: 256 -> 512, stay at 32x32
        model4 = [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True), ]
        model4 += [nn.ReLU(True), ]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), ]
        model4 += [nn.ReLU(True), ]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), ]
        model4 += [nn.ReLU(True), ]
        model4 += [norm_layer(512), ]

        # Model 5: Dilated convolutions
        model5 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model5 += [nn.ReLU(True), ]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model5 += [nn.ReLU(True), ]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model5 += [nn.ReLU(True), ]
        model5 += [norm_layer(512), ]

        # Model 6: Dilated convolutions
        model6 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model6 += [nn.ReLU(True), ]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model6 += [nn.ReLU(True), ]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model6 += [nn.ReLU(True), ]
        model6 += [norm_layer(512), ]

        # Model 7: Standard convolutions
        model7 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), ]
        model7 += [nn.ReLU(True), ]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), ]
        model7 += [nn.ReLU(True), ]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), ]
        model7 += [nn.ReLU(True), ]
        model7 += [norm_layer(512), ]

        # Model 8: Decoder
        model8 = [nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True), ]
        model8 += [nn.ReLU(True), ]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), ]
        model8 += [nn.ReLU(True), ]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), ]
        model8 += [nn.ReLU(True), ]
        model8 += [nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True), ]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, input_l):
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))

        return self.unnormalize_ab(self.upsample4(out_reg))


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator for 70x70 patches
    Input: L channel (1) + ab channels (2) = 3 channels
    """

    def __init__(self, in_channels=3):
        super(PatchGANDiscriminator, self).__init__()

        def discriminator_block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # No norm in first layer
            *discriminator_block(in_channels, 64, normalize=False),  # 128x128
            *discriminator_block(64, 128),  # 64x64
            *discriminator_block(128, 256),  # 32x32
            *discriminator_block(256, 512),  # 16x16
            # Final layer - no stride
            nn.Conv2d(512, 1, 4, padding=1)  # 30x30 output
        )

    def forward(self, img_L, img_ab):
        # Concatenate L and ab channels
        img_input = torch.cat([img_L, img_ab], dim=1)
        return self.model(img_input)


def resize_img(img, HW=(256, 256), resample=Image.BILINEAR):
    """
    Resize numpy image to (HW[0], HW[1]).
    """
    return np.asarray(
        Image.fromarray(img).resize((HW[1], HW[0]), resample=resample)
    )


def preprocess_img(img_rgb_orig, HW=(256, 256), resample=Image.BILINEAR, return_ab=False):
    """
    Preprocess image into L_orig (original size) and L_rs (resized).
    Returns:
      tens_orig_l : (1,1,H_orig,W_orig) -> original L channel tensor
      tens_rs_l   : (1,1,HW[0],HW[1]) -> resized L channel tensor
    """
    # Resize original
    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)

    # Convert both original & resized images to LAB
    img_lab_orig = color.rgb2lab(img_rgb_orig)
    img_lab_rs = color.rgb2lab(img_rgb_rs)

    # Extract only L channel
    img_l_orig = img_lab_orig[:, :, 0]
    img_l_rs = img_lab_rs[:, :, 0]

    # Convert to torch tensors with shape (1,1,H,W)
    tens_orig_l = torch.tensor(img_l_orig, dtype=torch.float32)[None, :, :]
    tens_rs_l = torch.tensor(img_l_rs, dtype=torch.float32)[None, :, :]

    #### Normalize L channel (paper recommends dividing by 100) -> I will look into this again(for now, lets have this) ####
    # tens_orig_l = tens_orig_l / 100.0
    # tens_rs_l   = tens_rs_l / 100.0

    # If pretrained is not True then we could center and Normalise the images
    # if pretrained == False:
    #     normalizer = BaseColor()
    #     tens_orig_l = normalizer.normalize_l(tens_orig_l)
    #     tens_rs_l   = normalizer.normalize_l(tens_rs_l)

    if return_ab:
        img_ab_rs = img_lab_rs[:, :, 1:]
        tens_ab = torch.tensor(img_ab_rs, dtype=torch.float32).permute(2, 0, 1)
        # tens_ab = normalizer.normalize_ab(tens_ab)

        # if not pretrained:
        #     tens_ab = normalizer.normalize_ab(tens_ab)

        return tens_rs_l, tens_ab

    return tens_orig_l, tens_rs_l


def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
    """
    Combine predicted ab channels with original L channel and convert back to RGB.
    """

    HW_orig = tens_orig_l.shape[1:]  # (H_orig, W_orig)
    HW_pred = out_ab.shape[1:]  # (H_pred, W_pred)

    # If needed, resize ab to match original size
    if HW_pred != HW_orig:
        out_ab = F.interpolate(out_ab.unsqueeze(0),
                               size=HW_orig, mode='bilinear')[0]

    # if not pretrained:
    #     normalizer = BaseColor()
    #     tens_orig_l = normalizer.unnormalize_l(tens_orig_l)
    # out_ab = normalizer.unnormalize_ab(out_ab)

    # tens_orig_l: (1, H_orig, W_orig)
    # out_ab:      (2, H_orig, W_orig)

    out_lab = torch.cat((tens_orig_l, out_ab), dim=0)  # (3, H_orig, W_orig)

    out_lab_np = out_lab.cpu().numpy().transpose((1, 2, 0))
    out_rgb = color.lab2rgb(out_lab_np)

    return out_rgb


# ------------------------------------------------
# DEFINE DATASET CLASS
# ------------------------------------------------

class ColorizationDataset(Dataset):
    """
    Dataset for automatic colorization.
    Loads images from folder, applies preprocessing,
    returns (L_resized, L_original, img_path).
    """

    def __init__(self, folder_path, pretrained=False, mode='inference'):
        self.image_paths = []
        self.pretrained = pretrained
        self.mode = mode

        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                self.image_paths.append(os.path.join(folder_path, fname))

        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]

        # Load image to numpy RGB
        img_rgb_orig = load_img(img_path)

        # Apply preprocessing
        if self.mode == 'training':
            # Get L and ab for training
            tens_l, tens_ab = preprocess_img(img_rgb_orig, HW=(IMAGE_SIZE, IMAGE_SIZE),
                                             return_ab=True)
            return tens_l, tens_ab
        else:
            # Get L only for inference
            tens_orig_l, tens_rs_l = preprocess_img(img_rgb_orig, HW=(IMAGE_SIZE, IMAGE_SIZE),
                                                    return_ab=False)
            return tens_rs_l, tens_orig_l, img_path

        # Return:
        # - resized L (input for model)
        # - original L (for reconstruction)
        # - image path


def colorization_collate(batch):
    """
    Allows batching resized L (fixed size),
    while keeping original L tensors in a list.
    """
    tens_rs_l_batch = torch.stack([b[0] for b in batch], dim=0)  # batchable
    tens_orig_l_list = [b[1] for b in batch]  # NOT stacked
    img_paths = [b[2] for b in batch]  # list of paths

    return tens_rs_l_batch, tens_orig_l_list, img_paths


# ------------------------------------------------
# DEFINE MODEL ARCHITECTURE & TRAINING LOOP
# ------------------------------------------------

def define_colorization_model(pretrained=True):
    """
    TODO:
    - Build U-Net / encoder-decoder model
    - Output: predicted ab channels (1,2,256,256)
    """
    model = ECCVGenerator()

    if pretrained:
        import torch.utils.model_zoo as model_zoo
        model.load_state_dict(
            model_zoo.load_url(
                'https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth',
                map_location='cpu',
                check_hash=True
            )
        )

    model.to(DEVICE)
    return model
    pass


def gan_loss(predictions, target_is_real, device):
    """
    GAN loss with label smoothing
    """
    if target_is_real:
        # Real labels with smoothing: 0.9 instead of 1.0
        target = torch.ones_like(predictions) * 0.9
    else:
        # Fake labels: 0.0
        target = torch.zeros_like(predictions)

    loss = F.binary_cross_entropy_with_logits(predictions, target)
    return loss


def calculate_losses(generator, discriminator, feature_extractor, criterion_content, real_L, real_ab, device):
    """
    Calculate all losses for one batch
    Returns: g_loss, d_loss, l1_loss
    """
    batch_size = real_L.size(0)

    # Generate fake ab channels
    fake_ab = generator(real_L)

    # ===== DISCRIMINATOR LOSS =====
    # Real images
    d_real = discriminator(real_L, real_ab)
    d_loss_real = gan_loss(d_real, True, device)

    # Fake images (detach to not train generator)
    d_fake = discriminator(real_L, fake_ab.detach())
    d_loss_fake = gan_loss(d_fake, False, device)

    # Total discriminator loss
    d_loss = (d_loss_real + d_loss_fake) * 0.5

    # ===== GENERATOR LOSS =====
    # Adversarial loss (try to fool discriminator)
    d_fake_for_g = discriminator(real_L, fake_ab)
    g_loss_gan = gan_loss(d_fake_for_g, True, device)

    # L1 loss (pixel-wise accuracy)
    l1_loss = F.l1_loss(fake_ab, real_ab)

    # Perceptual loss
    fake_lab = torch.cat([real_L, fake_ab], dim=1)  # (B, 3, H, W)
    real_lab = torch.cat([real_L, real_ab], dim=1)
    gen_features = feature_extractor(fake_lab)
    real_features = feature_extractor(real_lab)
    loss_content = criterion_content(gen_features, real_features.detach())

    # Combined generator loss
    g_loss = g_loss_gan + LAMBDA_L1 * l1_loss + loss_content

    return g_loss, d_loss, l1_loss, g_loss_gan


def save_sample_images(generator, data_loader, epoch, save_dir="gan_samples"):
    """
    Save ONE sample image per epoch to monitor progress
    """
    os.makedirs(save_dir, exist_ok=True)

    generator.eval()
    with torch.no_grad():
        # Get one batch
        for tens_l, tens_ab in data_loader:
            tens_l = tens_l.to(DEVICE)
            tens_ab = tens_ab.to(DEVICE)

            # Generate fake colors
            fake_ab = generator(tens_l)

            # Save ONLY first image
            l_channel = tens_l[0].cpu()
            real_ab = tens_ab[0].cpu()
            pred_ab = fake_ab[0].cpu()

            # Create side-by-side comparison
            gray_rgb = postprocess_tens(l_channel, torch.zeros_like(real_ab))
            real_rgb = postprocess_tens(l_channel, real_ab)
            fake_rgb = postprocess_tens(l_channel, pred_ab)

            # Concatenate horizontally
            comparison = np.concatenate([gray_rgb, real_rgb, fake_rgb], axis=1)

            # Save
            save_path = os.path.join(save_dir, f"epoch_{epoch}.png")
            Image.fromarray((comparison * 255).astype(np.uint8)).save(save_path)

            break  # Only process one batch

    generator.train()
    print(f"✓ Saved sample image: {save_path}")


def run_inference(generator, data_folder, output_folder="gan_inference", num_images=50):
    """
    Run inference on 50 random images after training
    """
    print(f"\nStarting inference on {num_images} images...")

    # Load dataset
    full_dataset = ColorizationDataset(data_folder, mode='inference')

    # Random sample
    indices = torch.randperm(len(full_dataset))[:num_images]
    inference_dataset = torch.utils.data.Subset(full_dataset, indices)

    inference_loader = DataLoader(
        inference_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=colorization_collate
    )

    os.makedirs(output_folder, exist_ok=True)

    generator.eval()
    with torch.no_grad():
        for batch_idx, (tens_rs_l, tens_orig_l, img_paths) in enumerate(inference_loader):
            tens_rs_l = tens_rs_l.to(DEVICE)
            predicted_ab = generator(tens_rs_l)

            for i in range(len(img_paths)):
                pred_ab = predicted_ab[i].cpu()
                orig_l = tens_orig_l[i]

                colorized_rgb = postprocess_tens(orig_l, pred_ab)

                filename = os.path.basename(img_paths[i])
                save_path = os.path.join(output_folder, f"gan_{filename}")
                Image.fromarray((colorized_rgb * 255).astype(np.uint8)).save(save_path)

        print(f"✓ Saved {num_images} images to {output_folder}/")


def train_gan(generator, data_folder, pretrained_path=None, save_interval=5):
    """
    Train GAN with pretrained ECCV generator
    """
    # Load pretrained weights if provided
    # print("Loading Zhang pretrained model (original)...")
    # import torch.utils.model_zoo as model_zoo
    # generator.load_state_dict(
    #     model_zoo.load_url(
    #         'https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth',
    #         map_location=DEVICE,
    #         check_hash=True
    #     )
    # )
    init_loss_log()

    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained model from: {pretrained_path}")
        generator.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))
        print(f"Loaded pretrained weights from {pretrained_path}")
    else:
        print("No pretrained weights loaded! Training from scratch.")

    # Initialize discriminator
    discriminator = PatchGANDiscriminator().to(DEVICE)
    feature_extractor = FeatureExtractor().to(DEVICE)
    feature_extractor.eval()

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=LR_G, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR_D, betas=(0.5, 0.999))
    criterion_content = nn.L1Loss()
    # Dataset
    train_dataset = ColorizationDataset(data_folder, mode='training')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)

    # Fixed validation sample (same image every epoch)
    val_dataset = ColorizationDataset(data_folder, mode='training')
    val_subset = torch.utils.data.Subset(val_dataset, [0])  # Just first image
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

    best_g_loss = float('inf')
    best_epoch = 0

    # Training loop
    for epoch in range(EPOCHS):
        generator.train()
        discriminator.train()

        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_l1_loss = 0
        epoch_gan_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for batch_idx, (tens_l, tens_ab) in enumerate(pbar):
            tens_l = tens_l.to(DEVICE)
            tens_ab = tens_ab.to(DEVICE)

            # Calculate losses
            g_loss, d_loss, l1_loss, g_gan_loss = calculate_losses(
                generator, discriminator, feature_extractor, criterion_content, tens_l, tens_ab, DEVICE
            )

            # ===== UPDATE DISCRIMINATOR =====
            optimizer_D.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            optimizer_D.step()

            # ===== UPDATE GENERATOR =====
            if batch_idx % D_STEPS == 0:
                optimizer_G.zero_grad()
                g_loss, _, l1_loss, g_gan_loss = calculate_losses(
                    generator, discriminator, feature_extractor, criterion_content, tens_l, tens_ab, DEVICE
                )
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                optimizer_G.step()

            # Track losses
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_l1_loss += l1_loss.item()
            epoch_gan_loss += g_gan_loss.item()

            # Update progress bar
            pbar.set_postfix({
                'G_loss': f'{g_loss.item():.3f}',
                'D_loss': f'{d_loss.item():.3f}',
                'L1': f'{l1_loss.item():.3f}',
                'GAN': f'{g_gan_loss.item():.3f}'
            })

        n_batches = len(train_loader)
        avg_g_loss = epoch_g_loss / n_batches
        avg_d_loss = epoch_d_loss / n_batches
        avg_l1_loss = epoch_l1_loss / n_batches
        avg_gan_loss = epoch_gan_loss / n_batches

        # Epoch summary
        n_batches = len(train_loader)
        print(f"\n[Epoch {epoch + 1}] "
              f"G_loss: {epoch_g_loss / n_batches:.4f} | "
              f"D_loss: {epoch_d_loss / n_batches:.4f} | "
              f"L1_loss: {epoch_l1_loss / n_batches:.4f} | "
              f"GAN_loss: {epoch_gan_loss / n_batches:.4f}")
        log_losses(epoch + 1, avg_g_loss, avg_d_loss, avg_l1_loss, avg_gan_loss)


        # Save ONE sample image per epoch
        save_sample_images(generator, val_loader, epoch + 1)

        # ===== SAVE BEST MODEL =====
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            best_epoch = epoch + 1
            torch.save(generator.state_dict(), 'best_gan_generator.pth')
            torch.save(discriminator.state_dict(), 'best_gan_discriminator.pth')
            print(f"✓ NEW BEST MODEL! Saved at epoch {best_epoch} with G_loss: {best_g_loss:.2f}")

        # Save checkpoints
        if (epoch + 1) % save_interval == 0:
            torch.save(generator.state_dict(),
                       f'gan_generator_epoch_{epoch + 1}.pth')
            torch.save(discriminator.state_dict(),
                       f'gan_discriminator_epoch_{epoch + 1}.pth')
            print(f"Saved checkpoint at epoch {epoch + 1}")

    return generator, discriminator


def plot_losses(csv_file='training_losses.csv', save_path='loss_plot.png'):
    """Plot and save training losses"""
    import pandas as pd

    # Read CSV
    df = pd.read_csv(csv_file)

    # Create plot
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(df['Epoch'], df['G_Loss'], 'b-', label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Loss')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(df['Epoch'], df['D_Loss'], 'r-', label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Discriminator Loss')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(df['Epoch'], df['L1_Loss'], 'g-', label='L1 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('L1 Loss')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(df['Epoch'], df['GAN_Loss'], 'm-', label='GAN Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Loss plot saved to {save_path}")

def train_colorization_model(
        model,
        data_folder,
        save_images=False,
        output_folder="training_output",
        finetune=True,
        inference_count=50
):
    """
    Training + inference pipeline.
    If finetune=False → skip training and run inference only.
    inference_count: number of random images to colorize.
    """

    # ------------------------------------------
    # TRAINING PHASE if finetune=True)
    # ------------------------------------------
    if finetune:
        print(f"Finetune = True , Starting training for {EPOCHS} epochs...")

        train_dataset = ColorizationDataset(data_folder, mode='training', pretrained=False)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        model.train()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.MSELoss()

        if save_images:
            os.makedirs(output_folder, exist_ok=True)

        for epoch in range(EPOCHS):
            running_loss = 0.0

            for batch_idx, (tens_l, tens_ab_gt) in enumerate(
                    tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
            ):
                tens_l = tens_l.to(DEVICE)
                tens_ab_gt = tens_ab_gt.to(DEVICE)

                optimizer.zero_grad()
                predicted_ab = model(tens_l)
                loss = criterion(predicted_ab, tens_ab_gt)
                loss.backward()
                optimizer.step()

                if epoch == 0 and batch_idx == 0:
                    print(
                        f"predicted_ab stats: min={predicted_ab.min():.2f}, "
                        f"max={predicted_ab.max():.2f}, mean={predicted_ab.mean():.2f}"
                    )
                    print(
                        f"tens_ab_gt stats: min={tens_ab_gt.min():.2f}, "
                        f"max={tens_ab_gt.max():.2f}, mean={tens_ab_gt.mean():.2f}"
                    )
                    print(f"Loss: {loss.item():.4f}")

                running_loss += loss.item()

                if save_images and batch_idx % 500 == 0:
                    with torch.no_grad():
                        for i in range(min(2, tens_l.shape[0])):
                            orig_l = tens_l[i].cpu()
                            pred_ab = predicted_ab[i].cpu()
                            colorized_rgb = postprocess_tens(orig_l, pred_ab)
                            save_path = os.path.join(
                                output_folder,
                                f"epoch{epoch + 1}_batch{batch_idx}_img{i}.png"
                            )
                            Image.fromarray((colorized_rgb * 255).astype(np.uint8)).save(save_path)

            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

        print("Training completed.")

        # Save fine tuned weights
        # torch.save(model.state_dict(), 'colorization_model_finetuned.pth')
        # print("Finetuned model saved as colorization_model_finetuned.pth")

    else:
        print("Finetune = False, Skipping training. Running inference only.")

    # ------------------------------------------
    # INFERENCE PHASE
    # ------------------------------------------
    print(f"\nStarting inference on: {data_folder}")

    # Load full inference dataset
    full_dataset = ColorizationDataset(data_folder, mode='inference', pretrained=False)

    # Limit inference to N random images
    if inference_count is not None:
        print(f"Sampling {inference_count} random images for inference...")
        indices = torch.randperm(len(full_dataset))[:inference_count]
        inference_dataset = torch.utils.data.Subset(full_dataset, indices)
    else:
        inference_dataset = full_dataset

    inference_loader = DataLoader(
        inference_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=colorization_collate
    )

    final_output = output_folder + "_final"
    os.makedirs(final_output, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for batch_idx, (tens_rs_l, tens_orig_l, img_paths) in enumerate(inference_loader):
            tens_rs_l = tens_rs_l.to(DEVICE)
            predicted_ab = model(tens_rs_l)

            for i in range(len(img_paths)):
                pred_ab = predicted_ab[i].cpu()
                orig_l = tens_orig_l[i]

                colorized_rgb = postprocess_tens(orig_l, pred_ab)

                filename = os.path.basename(img_paths[i])
                save_path = os.path.join(final_output, f"colorized_{filename}")
                Image.fromarray((colorized_rgb * 255).astype(np.uint8)).save(save_path)

            print(f"Batch {batch_idx + 1}/{len(inference_loader)} colorized")

    print(f"Saved {len(inference_dataset)} images → {final_output}")

    return model


# ------------------------------------------------
# MAIN BLOCK
# ------------------------------------------------

# ------------------------------------------------
# MAIN BLOCK
# ------------------------------------------------

if __name__ == "__main__":
    generator = ECCVGenerator()
    generator.to(DEVICE)

    # Train GAN
    print("STARTING GAN TRAINING")
    trained_gen, trained_disc = train_gan(
        generator=generator,
        data_folder="imagenet_50/train",
        pretrained_path="colorization_model.pth",
        save_interval=5
    )

    # Load BEST model for inference
    print("\nLoading BEST model for inference...")
    best_generator = ECCVGenerator()
    best_generator.load_state_dict(torch.load('best_gan_generator.pth', map_location=DEVICE))
    best_generator.to(DEVICE)

    # Run inference with BEST model
    print("RUNNING INFERENCE WITH BEST MODEL")
    run_inference(
        generator=best_generator,
        data_folder="imagenet_50/train",
        output_folder="gan_colorized_output",
        num_images=50
    )

    plot_losses('training_losses.csv', 'training_losses.png')
