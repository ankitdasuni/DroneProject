import cv2
import numpy as np
#this is a project for iitr tech fest 
# Function to align images using SIFT keypoints and descriptors
def align_images(img1, img2):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors in the two images
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Initialize brute force matcher
    bf = cv2.BFMatcher()

    # Match descriptors
    matches = bf.knnMatch(des1, des2, k=2)

    # Filter good matches using Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Minimum number of matches required
    min_matches = 10
    if len(good_matches) > min_matches:
        # Get matching keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Estimate perspective transformation
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Warp image 1 to image 2's perspective
        aligned_img = cv2.warpPerspective(img1, M, (img2.shape[1] + img1.shape[1], img2.shape[0]))
        aligned_img[:, :img2.shape[1]] = img2
        return aligned_img
    else:
        print("Not enough matches to align images.")
        return None

# Function to stitch images
def stitch_images(images):
    # Initialize stitcher
    stitcher = cv2.Stitcher_create()
    # Try to stitch images
    status, stitched_image = stitcher.stitch(images)
    # If successful, return stitched image
    if status == cv2.Stitcher_OK:
        return stitched_image
    else:
        print("Error during stitching:", status)
        return None

# List of images to stitch
# image_paths = ["IMG20240211194551.jpg","image/IMG20240211194554.jpg","image/IMG20240211194556.jpg","image/IMG20240211194559.jpg"]
image_paths = ["image/2024-02-11-145604.jpg","image/2024-02-11-145605.jpg"]


# Read images
images = []
for path in image_paths:
    image = cv2.imread(path)
    if image is not None:
        images.append(image)

# Align images

aligned_images = [images[0]]
for i in range(1, len(images)):
    aligned_img = align_images(aligned_images[-1], images[i])
    if aligned_img is not None:
        aligned_images.append(aligned_img)


# Stitch aligned images
stitched_image = stitch_images(aligned_images)

# Display result
cv2.resize(stitched_image, (20,20))
if stitched_image is not None:
    cv2.imshow("Stitched Image", stitched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Stitching failed.")
