import random

import numpy
import torch
import numpy as np
from scipy.spatial import distance


class Circ:
    def __init__(self, circle_center_x, rec_long_side, rec_short_side, seed):
        self.circle_center_x = circle_center_x
        self.rec_long_side = rec_long_side
        self.rec_short_side = rec_short_side
        self.circle_radius = distance.euclidean([circle_center_x, 0], [rec_long_side / 2, rec_short_side / 2])
        self.seed = seed

    def create_samples(self, points_per_unit=100):
        diff_x = self.circle_center_x - (self.rec_long_side / 2)
        assert diff_x > 0

        add_angle = 1 / (points_per_unit * self.circle_radius)
        full_circles_perimeter = 4 * np.pi * self.circle_radius
        eye_samples = torch.empty((int(points_per_unit * full_circles_perimeter), 2), dtype=torch.float32)

        full_circle_sample = int(full_circles_perimeter * points_per_unit / 2)

        for i in range(full_circle_sample):
            eye_samples[i] = torch.Tensor(
                [-self.circle_center_x + self.circle_radius * np.cos(i * add_angle),
                 self.circle_radius * np.sin(i * add_angle)])
            eye_samples[full_circle_sample + i] = torch.Tensor(
                [self.circle_center_x + self.circle_radius * np.cos(i * add_angle),
                 self.circle_radius * np.sin(i * add_angle)])

        return eye_samples

    def subsample_shape(self, points_per_unit, percent_to_keep, special_selection=False):
        eye_samples = self.create_samples(points_per_unit)
        (dm, dn) = eye_samples.shape
        num_eye_subsample = int(percent_to_keep*dm)

        rand_eye_idx = torch.randperm(dm)
        if special_selection:
            left_sample_start = int((dm - num_eye_subsample) / 4)
            left_sample_end = int((dm + num_eye_subsample) / 4)
            left_sample_range = np.arange(left_sample_start, left_sample_end)
            right_sample1_start = int(dm / 2)
            right_sample1_end = int(dm / 2 + num_eye_subsample / 4)
            right_sample1_range = np.arange(right_sample1_start, right_sample1_end)
            right_sample2_start = int(dm - num_eye_subsample / 4)
            right_sample2_end = dm
            right_sample2_range = np.arange(right_sample2_start, right_sample2_end)
            right_sample_range = np.concatenate((right_sample1_range, right_sample2_range))
            eye_subsample_indices = np.concatenate((left_sample_range, right_sample_range))
            eye_remaining_indices = np.setdiff1d(np.arange(dm), eye_subsample_indices)
        else:
            eye_subsample_indices = rand_eye_idx[:num_eye_subsample]
            eye_remaining_indices = rand_eye_idx[num_eye_subsample:]

        return eye_samples[eye_remaining_indices, :], eye_samples[eye_subsample_indices, :]

    def create_negative_instances(self, positive_samples, bias):
        negative_samples = torch.empty(positive_samples.size(), dtype=torch.float32)
        for i in range(len(positive_samples)):
            x, y = positive_samples[i]
            negative_samples[i] = torch.Tensor(
                [x + random.choice([-1, 1]) * np.random.uniform(bias, 1) * self.rec_short_side / 2,
                 y + random.choice([-1, 1]) * np.random.uniform(bias, 1) * self.rec_short_side / 2])
        return negative_samples

    def create_dataset(self, train_instances=1000, test_instances=1, ppu_choices=None, percent_choices=None,
                       special_entry=10, bias=0.1):
        full_clouds_positive = []
        full_clouds_negative = []
        partial_clouds = []
        labels = []
        partial_clouds_test = []
        labels_test = []
        for i in range(train_instances):
            if ppu_choices is not None:
                ppu = random.choice(ppu_choices)
            else:
                ppu = random.choice([10, 25, 50, 100, 150, 200])
            if percent_choices is not None:
                percent = random.choice(percent_choices)
            else:
                percent = random.choice([0.05, 0.1, 0.15, 0.2, 0.25])
            if i % special_entry == 0:
                ds, dss = self.subsample_shape(ppu, percent, True)
            else:
                ds, dss = self.subsample_shape(ppu, percent)
            full_clouds_positive.append(ds.numpy())
            full_clouds_negative.append(self.create_negative_instances(ds, bias).numpy())
            labels.append(np.array([1, 0]))
            partial_clouds.append(dss.numpy())

        for j in range(test_instances):
            if ppu_choices is not None:
                ppu = random.choice(ppu_choices)
            else:
                ppu = random.choice([10, 25, 50, 100, 150, 200])
            if percent_choices is not None:
                percent = random.choice(percent_choices)
            else:
                percent = random.choice([0.05, 0.1, 0.15, 0.2, 0.25])
            if j % special_entry == 0:
                ds, dss = self.subsample_shape(ppu, percent, True)
            else:
                ds, dss = self.subsample_shape(ppu, percent)
            partial_clouds_test.append(dss.numpy())
            labels_test.append(np.array([1, 0]))

        return np.array(full_clouds_positive), np.array(full_clouds_negative), np.array(partial_clouds), np.array(labels), np.array(
            partial_clouds_test), np.array(labels_test)
