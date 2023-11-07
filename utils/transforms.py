import cv2
import numpy as np
import random


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, labels):
        for t in self.transforms:
            image, labels = t(image, labels)
        return image, labels


class RandomCrop(object):
    def __init__(self, jitter=0.3, resize=1.5, net_size=416):
        self.jitter = jitter
        self.resize = resize
        self.lowest_w = 1.0 / net_size
        self.lowest_h = 1.0 / net_size

    def rand_precalc_random(self, min, max, random_part):
        if max < min:
            swap = min
            min = max
            max = swap
        return int((random_part * (max - min)) + min)

    def __call__(self, image, labels):
        oh, ow, _ = image.shape
        dw = int(ow * self.jitter)
        dh = int(oh * self.jitter)

        resize_down, resize_up = self.resize, self.resize
        if resize_down > 1.0:
            resize_down = 1. / resize_down
        min_rdw = ow * (1 - (1. / resize_down)) // 2
        min_rdh = oh * (1 - (1. / resize_down)) // 2

        if resize_up < 1.0:
            resize_up = 1. / resize_up
        max_rdw = ow * (1 - (1. / resize_up)) // 2
        max_rdh = oh * (1 - (1. / resize_up)) // 2

        resize_r1 = random.uniform(0, 1)
        resize_r2 = random.uniform(0, 1)
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
        r3 = random.uniform(0, 1)
        r4 = random.uniform(0, 1)

        pleft = self.rand_precalc_random(-dw, dw, r1)
        pright = self.rand_precalc_random(-dw, dw, r2)
        ptop = self.rand_precalc_random(-dh, dh, r3)
        pbot = self.rand_precalc_random(-dh, dh, r4)

        pleft += self.rand_precalc_random(min_rdw, max_rdw, resize_r1)
        pright += self.rand_precalc_random(min_rdw, max_rdw, resize_r2)
        ptop += self.rand_precalc_random(min_rdh, max_rdh, resize_r1)
        pbot += self.rand_precalc_random(min_rdh, max_rdh, resize_r2)

        swidth = ow - pleft - pright
        sheight = oh - ptop - pbot

        cropped = self.crop_image(image, pleft, ptop, swidth, sheight)
        crop_h, crop_w, _ = cropped.shape

        labels_out = labels.copy()
        shift = np.array([pleft, ptop, pleft, ptop])  # [m, 4]
        labels_out[:, 1:] = labels_out[:, 1:] - shift
        labels_out[:, 1] = labels_out[:, 1].clip(min=0, max=crop_w)
        labels_out[:, 2] = labels_out[:, 2].clip(min=0, max=crop_h)
        labels_out[:, 3] = labels_out[:, 3].clip(min=0, max=crop_w)
        labels_out[:, 4] = labels_out[:, 4].clip(min=0, max=crop_h)

        mask_w = ((labels_out[:, 3] - labels_out[:, 1]) / crop_w > self.lowest_w)
        mask_h = ((labels_out[:, 4] - labels_out[:, 2]) / crop_h > self.lowest_h)
        labels_out = labels_out[mask_w & mask_h]

        if len(labels_out) == 0:
            return image, labels

        return cropped, labels_out

    def intersect_rect(self, rect1, rect2):
        x = max(rect1[0], rect2[0])
        y = max(rect1[1], rect2[1])
        width = min(rect1[0] + rect1[2], rect2[0] + rect2[2]) - x
        height = min(rect1[1] + rect1[3], rect2[1] + rect2[3]) - y
        return [x, y, width, height]

    def crop_image(self, img, pleft, ptop, swidth, sheight):
        oh, ow, _ = img.shape
        if pleft == 0 and ptop == 0 and swidth == ow and sheight == oh:
            return img
        else:
            src_rect = [pleft, ptop, swidth, sheight]
            img_rect = [0, 0, ow, oh]
            new_src_rect = self.intersect_rect(src_rect, img_rect)
            assert new_src_rect[2] > 0 and new_src_rect[3] > 0, 'no intersect'
            dst_rect = [max(0, -pleft), max(0, -ptop), new_src_rect[2], new_src_rect[3]]

            img_mean = cv2.mean(img)
            cropped_img = np.empty((sheight, swidth, 3), dtype=np.uint8)
            cropped_img[:, :] = img_mean[:3]
            cropped_img[dst_rect[1] : dst_rect[1] + dst_rect[3], dst_rect[0] : dst_rect[0] + dst_rect[2], :] = \
                img[new_src_rect[1] : new_src_rect[1] + new_src_rect[3], new_src_rect[0] : new_src_rect[0] + new_src_rect[2], :]

            return cropped_img


class RandomFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, labels):
        _, width, _ = image.shape
        if random.random() < self.prob:
            image = np.ascontiguousarray(image[:, ::-1])
            labels_cp = labels.copy()
            labels[:, 1::2] = width - labels_cp[:, 3::-2]

        return image, labels


class RandomHue(object):
    def __init__(self, hue=0.1, prob=0.5):
        self.hue = hue
        self.prob = prob

    def __call__(self, image, labels):
        if random.random() < self.prob:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            dhue = random.uniform(-self.hue, self.hue)
            h = h + 179 * dhue
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return image, labels


class RandomSaturation(object):
    def __init__(self, sat=1.5, prob=0.5):
        self.sat = sat
        self.prob = prob

    def __call__(self, image, labels):
        if random.random() < self.prob:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            scale = random.uniform(1., self.sat)
            if random.randrange(2):
                dsat = scale
            else:
                dsat = 1. / scale
            s = s * dsat
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return image, labels


class RandomExposure(object):
    def __init__(self, exp=1.5, prob=0.5):
        self.exp = exp
        self.prob = prob

    def __call__(self, image, labels):
        if random.random() < self.prob:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            scale = random.uniform(1., self.exp)
            if random.randrange(2):
                dexp = scale
            else:
                dexp = 1. / scale
            v = v * dexp
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return image, labels