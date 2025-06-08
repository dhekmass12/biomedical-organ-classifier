import os
import argparse
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


class MultiViewTestDataset(Dataset):
    """
    Dataset for multi-view images in the test set.
    Loads three views for each ID and returns a stacked 3-channel image.
    """
    def __init__(self, df, image_dir, class_map, transform=None):
        """
        df: DataFrame with column 'id' for test samples.
        image_dir: root directory containing 'test' folder with images.
        class_map: dict mapping class label to index (loaded from train).
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        # Invert class_map for index->label
        self.idx_to_class = {v: k for k, v in class_map.items()}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id = self.df.loc[idx, 'id']
        base = img_id.split('_', 1)[1]
        paths = [
            os.path.join(self.image_dir, 'test', f'a_{base}'),
            os.path.join(self.image_dir, 'test', f'c_{base}'),
            os.path.join(self.image_dir, 'test', f's_{base}')
        ]
        images = []
        for path in paths:
            if os.path.exists(path):
                img = Image.open(path).convert('L')
            else:
                img = Image.new('L', (128, 128))
            images.append(img)
        multi = Image.merge('RGB', (images[0], images[1], images[2]))
        if self.transform:
            multi = self.transform(multi)
        return multi, img_id

def predict(args):
    # Load the saved class mapping and best model weights
    class_map = torch.load(args.class_map)
    num_classes = len(class_map)
    # Same model architecture as in training, could use additional separate file for model def
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes):
            super(SimpleCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(128 * 8 * 8, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(256, num_classes)
            )
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    device = torch.device(args.device)
    model = SimpleCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    
    # Load test sample IDs
    df_test = pd.read_csv(os.path.join(args.data_dir, 'test.csv'))
    transform_test = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])
    test_dataset = MultiViewTestDataset(df_test, args.data_dir, class_map, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Perform inference
    ids = []
    preds = []
    for inputs, img_ids in tqdm(test_loader, desc="Running Inference"):
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, pred_indices = torch.max(outputs, 1)
        for i, img_id in enumerate(img_ids):
            ids.append(img_id)
            preds.append(test_dataset.idx_to_class[pred_indices[i].item()])
    
    submission = pd.DataFrame({'id': ids, 'class': preds})
    submission.to_csv('submission.csv', index=False)
    print("Saved predictions to submission.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict with trained multi-view CNN')
    parser.add_argument('--data_dir', type=str, default='organ-all-mnist-DL20242025',
                        help='Root dataset directory containing train/ and test/')
    parser.add_argument('--model', type=str, default='cnn_best.pth',
                        help='Path to trained model weights (.pth)')
    parser.add_argument('--class_map', type=str, default='cnn_class_map.pth',
                        help='Path to saved class map (from training)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for test loader')
    parser.add_argument('--device', type=str, default='cpu', help='Device for inference (cpu or cuda)')
    args = parser.parse_args()
    predict(args)
