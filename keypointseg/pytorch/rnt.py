import torch
import numpy
import cv2
import sys
from sklearn.cluster import DBSCAN

import time

#
from segnets import TinySegNet
model = TinySegNet(3, 3)
model.load_state_dict( torch.load('caltechfaces/params/2048.pth') )
model.cuda()

#
def cluster_pts(pts, eps=10, min_samples=32):
	#
	if pts.shape[0] == 0:
		return []
	#
	dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
	core_samples_mask = numpy.zeros_like(dbscan.labels_, dtype=bool)
	core_samples_mask[dbscan.core_sample_indices_] = True
	labels = dbscan.labels_
	n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
	#print(n_clusters)
	#print(labels)
	clusters = []
	for i in range(0, n_clusters):
		clusters.append( numpy.mean(pts[labels==i, :], 0) )
	return clusters

def get_bboxes(anchor, other, angle, tol):
	bboxes = []
	for a in other:
		for b in anchor:
			#
			v = b-a
			t = numpy.arctan2(v[0], v[1])*180.0/numpy.pi
			#print(t)
			if t>(angle-tol) and t<(angle+tol):
				s = numpy.linalg.norm(v)
				#print(s)
				#if s < 150:
				bboxes.append((b[0], b[1], s))
	return bboxes

def process_image(img, thr=0.1):
	#
	start = time.time()
	img = torch.from_numpy(img).cuda().permute(2, 0, 1).float().div(255.0)
	lbl = model( torch.autograd.Variable(img, volatile=True).unsqueeze(0) ).data[0].cpu()
	lbl = lbl.mul(lbl.gt(thr).float())
	print('* (1) ' + str(time.time() - start))
	#
	#return [], lbl.sum(0).mul(255).byte().numpy()
	return [], lbl.mul(255).byte().permute(1, 2, 0).numpy()

#
if len(sys.argv)>=2:
	#
	img = cv2.imread(sys.argv[1])
	maxhw = 160
	if img.shape[0]>maxhw or img.shape[1]>maxhw:
		scalefactor = numpy.min((maxhw/img.shape[0], maxhw/img.shape[1]))
	img = cv2.resize(img, (0,0), fx=scalefactor, fy=scalefactor)

	bboxes, lbl = process_image(img, thr=0.25)

	for b in bboxes:
		cv2.circle(img, (int(b[1]), int(b[0])), int(b[2]), (0, 0, 255), 4)
	cv2.imwrite('img.jpg', img)
	cv2.imwrite('lbl.jpg', lbl)
else:
	#
	cap = cv2.VideoCapture(0)
	while(True):
		#
		ret, frm = cap.read()
		frm = cv2.resize(frm, (0,0), fx=0.25, fy=0.25)
		#
		bboxes, sgm = process_image(frm, thr=0.25)
		for b in bboxes:
			cv2.circle(frm, (int(b[1]), int(b[0])), int(b[2]), (0, 0, 255), 4)
		#
		cv2.imshow('frm', frm)
		cv2.imshow('sgm', sgm)
		#
		if cv2.waitKey(5) & 0xFF == ord('q'):
			break
	#
	cap.release()
	cv2.destroyAllWindows()
