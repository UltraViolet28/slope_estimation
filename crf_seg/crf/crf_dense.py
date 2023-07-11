import numpy as np
import pydensecrf.densecrf as dcrf

def compute_pairwise_potentials(point_cloud, surface_normals):
    # Compute pairwise potentials based on surface normals
    # Here, you can define a similarity measure based on the dot product or angle between normals
    # Construct a similarity matrix based on the pairwise comparisons
    similarity_matrix = np.dot(surface_normals, surface_normals.T)
    
    # Normalize the similarity matrix to obtain pairwise potentials between 0 and 1
    pairwise_potentials = (similarity_matrix - np.min(similarity_matrix)) / (np.max(similarity_matrix) - np.min(similarity_matrix))
    
    return pairwise_potentials

def crf_inference(unary_potentials, pairwise_potentials):
    # Perform CRF inference using dense CRF
    num_classes = unary_potentials.shape[1]
    num_points = unary_potentials.shape[0]

    # Create a dense CRF model
    crf = dcrf.DenseCRF(num_points, num_classes)
    
    # Set unary potentials
    unary = unary_potentials.transpose(1, 0)  # CRF expects shape (num_classes, num_points)
    crf.setUnaryEnergy(-np.log(unary))
    
    # Set pairwise potentials
    pairwise = pairwise_potentials.transpose(2, 0, 1)  # CRF expects shape (num_classes, num_classes, num_points)
    crf.addPairwiseEnergy(pairwise, compat=10)
    
    # Perform inference
    infer_labels = crf.inference(5)  # Run CRF inference for 5 iterations
    
    # Reshape inferred labels to original shape (num_points, num_classes)
    inferred_segmentation = np.array(infer_labels).reshape((num_classes, num_points)).transpose(1, 0)
    
    return inferred_segmentation


class CRF_layer(nn.Module):
    def __init__(self, num_classes, num_points):
        super(CRF_layer, self).__init__()
        self.num_classes = num_classes
        self.num_points = num_points
        self.crf = dcrf.DenseCRF(num_points, num_classes)
        
    def forward(self, unary_potentials, pairwise_potentials):
        unary = unary_potentials.transpose(1, 0)  # CRF expects shape (num_classes, num_points)
        self.crf.setUnaryEnergy(-np.log(unary))
        
        pairwise = pairwise_potentials.transpose(2, 0, 1)  # CRF expects shape (num_classes, num_classes, num_points)
        self.crf.addPairwiseEnergy(pairwise, compat=10)
        
        infer_labels = self.crf.inference(5)  # Run CRF inference for 5 iterations
        
        inferred_segmentation = np.array(infer_labels).reshape((self.num_classes, self.num_points)).transpose(1, 0)
        
        return inferred_segmentation

