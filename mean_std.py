import json
import cv2
import os
import numpy as np

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def compute_mean_std(path, path_prefix, key1, key2):
	"""
	compute mean and std with annotation json file.
	eg.
	{'info':'', 'images':{"file_name": "iamge_1.jpg"}}
	"""
	with open(path, 'r') as f:
		annotations = json.load(f)
	images_info = annotations[key1]
	count = 0
	mean_r = 0
	mean_g = 0
	mean_b = 0
	for image_info in images_info:
		name = os.path.join(path_prefix, image_info[key2])
		if  not os.path.exists(name):
			continue
		count += 1
		#print(name)
		image = cv2.imread(name)
		mean_b += np.mean(image[:, :, 0])
		mean_g += np.mean(image[:, :, 1])
		mean_r += np.mean(image[:, :, 2])

	assert count > 0
	mean_b /= count
	mean_g /= count
	mean_r /= count

	diff_r = 0
	diff_g = 0
	diff_b = 0

	N = 0
	for image_info in images_info:
		name = os.path.join(path_prefix, image_info[key2])
		if  not os.path.exists(name):
			continue
		count += 1
		image = cv2.imread(name)
		diff_b += np.sum(np.power(image[:, :, 0] - mean_b, 2))
		diff_g += np.sum(np.power(image[:, :, 1] - mean_g, 2))
		diff_r += np.sum(np.power(image[:, :, 2] - mean_r, 2))
		N += np.prod(image[:, :, 0].shape)
		
	assert N > 0
	std_b = np.sqrt(diff_b / N)
	std_g = np.sqrt(diff_g / N)
	std_r = np.sqrt(diff_r / N)

	mean = (mean_b.item() / 255.0, mean_g.item() / 255.0, mean_r.item() / 255.0)
	std = (std_b.item() / 255.0, std_g.item() / 255.0, std_r.item() / 255.0)
	return mean, std


if __name__ == '__main__':
	prefix = os.path.join(os.getcwd(), 'images') 
	mean, std = compute_mean_std('annotations.json', prefix, 'images', 'file_name')
	print(mean)
	print(std)
						


			