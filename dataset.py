from torch.utils.data import Dataset
import cv2
import torchvision.transforms as transforms
import glob


normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.9357, 0.8253, 0.8998), (0.0787, 0.1751, 0.1125)),
])


class TMADataset(Dataset):
    """PyTorch dataset - loading jpg images from a root directory."""

    def __init__(self,
                 images_root_dir: str,
                 ):
        super(TMADataset, self).__init__()
        self.images_root_dir = images_root_dir
        self.target_size = 512
        self.images = glob.glob(images_root_dir + '/**/*.jpg', recursive=True)
        print(f'Found {len(self.images)} .jpg images')

    def __len__(self):
        """Return no. of samples."""
        return len(self.images)

    def __getitem__(self, index: int):
        im_path = self.images[index]
        with open(im_path, 'rb') as f:
            check_chars = f.read()[-2:]
        if check_chars != b'\xff\xd9':
            raise Exception('Error while loading JPEG image!')

        im = cv2.imread(im_path)
        if im is None:
            raise Exception(f'Error in loading file {im_path}')

        h, w, c = im.shape

        center_margin = (w - h) // 2
        if center_margin > 0:
            im = im[:, center_margin:-center_margin]
        elif center_margin < 0:
            im = im[-center_margin:center_margin, :]

        im = cv2.resize(im, dsize=(self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        im = normalize(im / 255)

        return im, im_path

