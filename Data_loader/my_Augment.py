import imgaug as ia
from imgaug import augmenters as iaa


def augment_img(image):
    # print('my_augment',image.shape)

    flipped_images = []

    flip_aug = iaa.Fliplr(1)
    flipped_images = flip_aug.augment_image(image)

    rotated_images = rotateImage(image,flipped_images)
    shifted_images = shiftImage(image,flipped_images)    
    scaled_images = scaleImage(image,flipped_images)

    augmented_images = [image, flipped_images, *rotated_images, *shifted_images, *scaled_images]

        # cv2.imshow("Image",images[i])
        # cv2.waitKey(0)
        # cv2.imshow("Flipped", flipped_images[i])
        # cv2.waitKey(0)
    return augmented_images

def rotateImage(images,flipped_images):
    angles = [ -15, 15]
    rotated_images = []
    for angle in angles:
        rotate = iaa.Affine(rotate = angle)
        rotated_images.append(rotate.augment_image(images))
        rotated_images.append(rotate.augment_image(flipped_images))

    # ia.imshow(np.hstack(rotated_images))
    return rotated_images

def shiftImage(images,flipped_images):
    val = [-20,20]
    vertice = ["x", "y"]
    shifted_images = []
    for i in range(2):
        for j in range(2):
            shift = iaa.Affine(translate_px={vertice[j]:(val[i])})
            shifted_images.append(shift.augment_image(images))
            shifted_images.append(shift.augment_image(flipped_images))


    # ia.imshow(np.hstack(images))
    return shifted_images

def scaleImage(images,flipped_images):
    scaled_images = []
    scale_fac = [0.8,1.2]
    for i in range(2):
        sacle = iaa.Affine(scale = scale_fac[i])
        scaled_images.append(sacle.augment_image(images))
        scaled_images.append(sacle.augment_image(flipped_images))

    # ia.imshow(np.hstack(Images))
    return scaled_images