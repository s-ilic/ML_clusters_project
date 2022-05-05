import random
import numpy as np

def IOU(ann, centroids):
    w, h = ann
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid

        if c_w >= w and c_h >= h:
            similarity = w*h/(c_w*c_h)
        elif c_w >= w and c_h <= h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape

    return np.array(similarities)

def avg_IOU(anns, centroids):
    n,d = anns.shape
    sum = 0.

    for i in range(anns.shape[0]):
        sum+= max(IOU(anns[i], centroids))

    return sum/n

def print_anchors(centroids, argv):
    out_string = ''

    anchors = centroids.copy()

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    strides = [8, 16, 32]

    out_string = "raw anchors:\n["
    for i in sorted_indices:
        out_string += "%.2f,%.2f, " % (
            anchors[i,0],
            anchors[i,1],
        )
    print(out_string[:-2] + ']')

    out_string = "strided anchors:\n["
    for ix, i in enumerate(sorted_indices):
        out_string += "%.2f,%.2f, " % (
            anchors[i,0] * argv['input_size'] / strides[ix // 3],
            anchors[i,1] * argv['input_size'] / strides[ix // 3],
        )
    print(out_string[:-2] + ']')

    out_string = "int strided anchors:\n["
    for ix, i in enumerate(sorted_indices):
        out_string += "%d,%d, " % (
            np.round(anchors[i,0] * argv['input_size'] / strides[ix // 3]),
            np.round(anchors[i,1] * argv['input_size'] / strides[ix // 3]),
        )
    print(out_string[:-2] + ']')


def run_kmeans(ann_dims, anchor_num):
    ann_num = ann_dims.shape[0]
    iterations = 0
    prev_assignments = np.ones(ann_num)*(-1)
    iteration = 0
    old_distances = np.zeros((ann_num, anchor_num))

    indices = [random.randrange(ann_dims.shape[0]) for i in range(anchor_num)]
    centroids = ann_dims[indices]
    anchor_dim = ann_dims.shape[1]

    while True:
        distances = []
        iteration += 1
        for i in range(ann_num):
            d = 1 - IOU(ann_dims[i], centroids)
            distances.append(d)
        distances = np.array(distances) # distances.shape = (ann_num, anchor_num)

        print("iteration {}: dists = {}".format(iteration, np.sum(np.abs(old_distances-distances))))

        #assign samples to centroids
        assignments = np.argmin(distances,axis=1)

        if (assignments == prev_assignments).all() :
            return centroids

        #calculate new centroids
        centroid_sums=np.zeros((anchor_num, anchor_dim), np.float)
        for i in range(ann_num):
            centroid_sums[assignments[i]]+=ann_dims[i]
        for j in range(anchor_num):
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j) + 1e-6)

        prev_assignments = assignments.copy()
        old_distances = distances.copy()

def _main_(argv):

    with open(argv['annot_path'] + 'train.txt', 'r') as f:
        lines = f.readlines()
    with open(argv['annot_path'] + 'valid.txt', 'r') as f:
        lines += f.readlines()

    # run k_mean to find the anchors
    annotation_dims = []
    for l in lines:
        lspl = l.split()
        for o in lspl[1:]:
            ospl = [int(n) for n in o.split(',')[:4]]
            xmin, ymin, xmax, ymax = ospl[:4]
            relative_w = (float(xmax) - float(xmin))/argv['img_size']
            relative_h = (float(ymax) - float(ymin))/argv['img_size']
            annotation_dims.append(tuple(map(float, (relative_w, relative_h))))

    annotation_dims = np.array(annotation_dims)
    centroids = run_kmeans(annotation_dims, argv['n_anchors'])

    # write anchors to file
    print('\naverage IOU for', argv['n_anchors'], 'anchors:', '%0.2f' % avg_IOU(annotation_dims, centroids))
    print_anchors(centroids, argv)

if __name__ == '__main__':

    args = {
        'annot_path':'/home/users/ilic/ML/ML_clusters_project/YOLOV3/runs/2048x0p396_ds1_mb1_nopad_nocl/',
        'n_anchors':9,
        'img_size':2048,
        'input_size':2048,
    }
    print("Args:")
    print(args)
    _main_(args)