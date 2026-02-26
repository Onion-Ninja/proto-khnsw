# code to load worldcoin npz normalized image:
import numpy as np
seg_path =  "/home/nishkal/datasets/iris_db/IITD_v1/worldcoin_outputs_npz/segmentation/1_L/1_L_1_seg.npz"
data = np.load(seg_path, allow_pickle=True)
predictions = data["predictions"]
index2class = dict(data["index2class"]) 

# print(index2class)
# print(predictions)

print("+++Segmentation+++")
print("Type:", type(predictions))
print("Dtype:", predictions.dtype)
print("Shape:", predictions.shape)
print("Min value:", predictions.min())
print("Max value:", predictions.max())

norm_path = "/home/nishkal/datasets/iris_db/IITD_v1/worldcoin_outputs_npz/normalized/1_L/1_L_1_norm.npz"
data = np.load(norm_path, allow_pickle=False)
normalized_image = data["normalized_image"]
normalized_mask  = data["normalized_mask"]

print("+++Normalized Images+++")
print("Type:", type(normalized_image))
print("Dtype:", normalized_image.dtype)
print("Shape:", normalized_image.shape)
print("Min value:", normalized_image.min())
print("Max value:", normalized_image.max())

print("+++Normalized Mask+++")
print("Type:", type(normalized_mask))
print("Dtype:", normalized_mask.dtype)
print("Shape:", normalized_mask.shape)

template_path = "/home/nishkal/datasets/iris_db/IITD_v1/worldcoin_outputs_npz/templates/1_L/1_L_1.npz"
data = np.load(template_path, allow_pickle=False)
iris_code= data["iris_code"]
mask_code  = data["mask_code"]

print("+++Iris Code+++")
print("Type:", type(iris_code))
print("Dtype:", iris_code.dtype)
print("Shape:", iris_code.shape)
print("Min value:", iris_code.min())
print("Max value:", iris_code.max())

print("+++Iris Code Mask+++")
print("Type:", type(mask_code))
print("Dtype:", mask_code.dtype)
print("Shape:", mask_code.shape)
print("Min value:", mask_code.min())
print("Max value:", mask_code.max())

