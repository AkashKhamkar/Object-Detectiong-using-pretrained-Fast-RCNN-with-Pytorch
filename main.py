from torchvision.models import detection
import numpy as np
import torch
import cv2
import pickle



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(DEVICE)

image_path = "pets.jpg"
confidenece_1 = 0.5
ph = "coco.pickle"

CLASSES = pickle.loads(open(ph,"rb").read())
COLORS = np .random.uniform(0, 255, size=(len(CLASSES), 3))

#print(CLASSES)

model = detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True,
	num_classes=len(CLASSES), pretrained_backbone=True).to(DEVICE)
model.eval()

image = cv2.imread(image_path)
orig = image.copy()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.transpose((2, 0, 1))

image = np.expand_dims(image, axis=0)
image = image / 255.0
image = torch.FloatTensor(image)

image = image.to(DEVICE)
detections = model(image)[0]

for i in range(0, len(detections["boxes"])):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections["scores"][i]

	# filter out weak detections by ensuring the confidence is
	# greater than the minimum confidence
	if confidence > confidenece_1:
		# extract the index of the class label from the detections,
		# then compute the (x, y)-coordinates of the bounding box
		# for the object
		idx = int(detections["labels"][i])
		box = detections["boxes"][i].detach().cpu().numpy()
		(startX, startY, endX, endY) = box.astype("int")

		# display the prediction to our terminal

		label = "{}: {:.2f}%".format(CLASSES[idx-1], confidence * 100)
		print("[INFO] {}".format(label))
		# draw the bounding box and label on the image
		cv2.rectangle(orig, (startX, startY), (endX, endY),
			COLORS[idx], 2)
		y = startY - 15 if startY - 15 > 15 else startY + 15

		cv2.putText(orig, label, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)



cv2.imshow("Output", orig)
cv2.waitKey(0)


