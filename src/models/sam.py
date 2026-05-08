import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

class SAMModel(torch.nn.Module):

    def __init__(self, model_type="vit_b", checkpoint_path="sam_vit_b_01ec64.pth"): # vit_b è il modello più leggero
        super(SAMModel, self).__init__()
        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.mask_generator = SamAutomaticMaskGenerator(self.model)
        self.predictor = SamPredictor(self.model)

    def forward(self, image): # metodo forward per generare tutte le maschere possibili dall'immagine
        masks = self.mask_generator.generate(image)
        return masks

    def predict(self, image, point_coords=None, point_labels=None): # il modello si aspetta un mio aiuto sotto forma di coordinate o bounding box + aiuto numerico (0 o 1)
        self.predictor.set_image(image)
        
        # Predice le maschere basate sui punti forniti
        if point_coords is not None and point_labels is not None:
            masks, _, _ = self.predictor.predict(point_coords=point_coords, point_labels=point_labels)
            return masks
        else:
            raise ValueError("Point coordinates and labels must be provided for prediction.")