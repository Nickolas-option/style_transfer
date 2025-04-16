import torch
import torchvision
from facenet_pytorch import MTCNN, InceptionResnetV1


class FaceDetector:
    """
    A class for detecting faces in images and extracting facial embeddings.
    
    This detector uses MTCNN for face detection and InceptionResnetV1 for generating
    facial embeddings that can be used for face recognition or comparison.
    """
    
    def __init__(self, device, optimal_image_size=None):
        """
        Initialize the face detector with detection and embedding models.
        
        Args:
            device: The device to run the models on (CPU or CUDA)
            optimal_image_size: Optional size to resize images for better detection results
        """
        # Initializes detector
        self.mtcnn = MTCNN(select_largest=False, device=device)
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval()

        self.optimal_image_size = optimal_image_size

        if optimal_image_size is not None:
            self.resize = torchvision.transforms.Resize(
                (self.optimal_image_size, self.optimal_image_size)
            )

    def get_coords_and_embeds(self, image):
        """
        Detect faces in an image and extract their coordinates and embeddings.
        
        Args:
            image: A tensor image to process
            
        Returns:
            None: If no faces are detected
            tuple: (bounding_box, embedding) where:
                - bounding_box: Coordinates [x1, y1, x2, y2] of the detected face
                - embedding: Feature vector representing the detected face
        """
        # Gets bounding boxed for one tensor image
        # Returns Nones for images without faces
        # Otherwise returns coords and ebmeddings for image

        h, w = image.shape[1], image.shape[2]

        # According to original facenet lib, we should reshape image to size (160, 160)
        # In order to get higher accuracy of detection

        if self.optimal_image_size is not None:
            image = self.resize(image)

        h1, w1 = image.shape[1], image.shape[2]

        image = image[:, h1 // 10 : h1 * 9 // 10, w1 // 10 : w1 * 9 // 10]

        forward_image = torch.permute(image, (1, 2, 0)) * 255.0

        image_cropped = self.mtcnn.detect(forward_image)

        if self.mtcnn(forward_image) is not None:
            image_embedding = self.resnet(self.mtcnn(forward_image).unsqueeze(0))

        bounding_box = image_cropped[0]

        if bounding_box is None:
            return None
        else:

            result_bounding_box = bounding_box[0]

            if self.optimal_image_size is not None:

                result_bounding_box[0] *= h / self.optimal_image_size
                result_bounding_box[2] *= h / self.optimal_image_size

                result_bounding_box[1] *= w / self.optimal_image_size
                result_bounding_box[3] *= w / self.optimal_image_size

            return result_bounding_box, image_embedding[0].detach().numpy()
