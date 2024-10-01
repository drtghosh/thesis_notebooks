import random

import numpy
import torch
import numpy as np
from scipy.spatial import distance


class DumbCirc:
    def __init__(self, circle_center_x, rec_long_side, rec_short_side, seed):
        self.circle_center_x = circle_center_x
        self.rec_long_side = rec_long_side
        self.rec_short_side = rec_short_side
        self.circle_radius = distance.euclidean([circle_center_x, 0], [rec_long_side / 2, rec_short_side / 2])
        self.seed = seed

    def create_samples(self, points_per_unit=100, equal_size=False):
        diff_x = self.circle_center_x - (self.rec_long_side / 2)
        assert diff_x > 0

        angle = np.arccos((self.circle_center_x - self.rec_long_side / 2.0) / self.circle_radius)
        add_angle = 1 / (points_per_unit * self.circle_radius)
        circle_perimeter = 4 * (np.pi - angle) * self.circle_radius
        rec_perimeter = 2 * self.rec_long_side
        total_perimeter = circle_perimeter + rec_perimeter

        dumbbell_samples = torch.empty((int(points_per_unit * total_perimeter), 2), dtype=torch.float32)
        samples_till_now = 0
        rec_sample = int(rec_perimeter * points_per_unit / 2)
        circle_sample = int(circle_perimeter * points_per_unit / 2)

        for i in range(rec_sample):
            dumbbell_samples[i] = torch.Tensor([-self.rec_long_side / 2 + i / points_per_unit,
                                                self.rec_short_side / 2])
        samples_till_now += rec_sample

        starting_angle = np.pi - angle
        for j in range(circle_sample):
            new_angle = starting_angle - j * add_angle
            dumbbell_samples[samples_till_now + j] = torch.Tensor(
                [self.circle_center_x + self.circle_radius * np.cos(new_angle),
                 self.circle_radius * np.sin(new_angle)])
        samples_till_now += circle_sample

        for i in range(rec_sample):
            dumbbell_samples[samples_till_now + i] = torch.Tensor(
                [self.rec_long_side / 2 - i / points_per_unit, -self.rec_short_side / 2])
        samples_till_now += rec_sample

        starting_angle = angle
        for j in range(circle_sample):
            new_angle = starting_angle + j * add_angle
            dumbbell_samples[samples_till_now + j] = torch.Tensor(
                [-self.circle_center_x + self.circle_radius * np.cos(new_angle),
                 self.circle_radius * np.sin(new_angle)])
        samples_till_now += circle_sample

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

        if equal_size:
            min_size = min(samples_till_now, eye_samples.shape[0])
            (dm, dn) = dumbbell_samples.shape
            (em, en) = eye_samples.shape
            torch.manual_seed(self.seed)
            rand_dumbbell_idx = torch.randperm(dm)
            rand_eye_idx = torch.randperm(em)
            dumbbell_subsample_indices = rand_dumbbell_idx[:min_size]
            eye_subsample_indices = rand_eye_idx[:min_size]
            return dumbbell_samples[dumbbell_subsample_indices, :], eye_samples[eye_subsample_indices, :]
        else:
            return dumbbell_samples[:samples_till_now], eye_samples

    def subsample_shape(self, points_per_unit, percent_to_keep, equal_size):
        dumbbell_samples, eye_samples = self.create_samples(points_per_unit, equal_size)
        (dm, dn) = dumbbell_samples.shape
        (em, en) = eye_samples.shape
        num_dumbbell_subsample = int(percent_to_keep*dm)
        num_eye_subsample = int(percent_to_keep*em)

        torch.manual_seed(self.seed)
        rand_dumbbell_idx = torch.randperm(dm)
        rand_eye_idx = torch.randperm(em)
        dumbbell_subsample_indices = rand_dumbbell_idx[:num_dumbbell_subsample]
        dumbbell_remaining_indices = rand_dumbbell_idx[num_dumbbell_subsample:]
        eye_subsample_indices = rand_eye_idx[:num_eye_subsample]
        eye_remaining_indices = rand_eye_idx[num_eye_subsample:]

        return (dumbbell_samples[dumbbell_remaining_indices, :], eye_samples[eye_remaining_indices, :],
                dumbbell_samples[dumbbell_subsample_indices, :], eye_samples[eye_subsample_indices, :])

    def subsample_eye_side(self, percent_to_keep=60, points_per_unit=100, equal_size=False):
        dumbbell_samples, eye_samples = self.create_samples(points_per_unit, equal_size)
        full_sample_size = eye_samples.size(0)
        half_sample_size = full_sample_size / 2
        throw = (100.0 - percent_to_keep) / 100.0
        half_throw = half_sample_size * throw / 2
        half_keep = half_sample_size * percent_to_keep / 200.0
        subsample_indices_l = np.arange(half_throw, half_sample_size - half_throw, 1)
        subsample_indices_r = np.concatenate(
            (half_sample_size + np.arange(half_keep), np.arange(full_sample_size - half_keep, full_sample_size, 1)))
        subsample_indices = np.concatenate((subsample_indices_l, subsample_indices_r))

        return eye_samples[subsample_indices, :]

    def create_negative_instances(self, positive_samples, bias):
        negative_samples = []
        for sample in positive_samples:
            x, y = sample
            negative_samples.append([x + random.choice([-1, 1]) * np.random.uniform(bias, 1) * self.rec_short_side / 2,
                                     y + random.choice([-1, 1]) * np.random.uniform(bias, 1) * self.rec_short_side / 2])
        return np.array(negative_samples)

    def create_dataset(self, train_instances=1000, test_instances=1, ppu_choices=None, percent_choices=None, equal_size=False, special_entry=10, bias=0.1):
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
            ds, es, dss, ess = self.subsample_shape(ppu, percent, equal_size)
            full_clouds_positive.append(ds.numpy())
            full_clouds_negative.append(self.create_negative_instances(ds.numpy(), bias))
            labels.append(np.array([1, 0]))
            full_clouds_positive.append(es.numpy())
            full_clouds_negative.append(self.create_negative_instances(es.numpy(), bias))
            labels.append(np.array([0, 1]))
            if i % special_entry == 0:
                special_partial = self.subsample_eye_side()
                tm, tn = special_partial.shape
                tm_new = tm
                if tm > len(dss):
                    tm_new = len(dss)
                    torch.manual_seed(self.seed)
                    rand_test_idx = torch.randperm(tm)
                    test_subsample_indices = rand_test_idx[:tm_new]
                    partial_to_add = special_partial[test_subsample_indices, :]
                else:
                    tm_needed = len(dss) - tm
                    torch.manual_seed(self.seed)
                    rand_needed_idx = torch.randperm(tm)
                    needed_indices = rand_needed_idx[:tm_needed]
                    partial_to_add = np.concatenate(special_partial, special_partial[needed_indices, :])
                partial_clouds.append(partial_to_add.numpy())
                partial_clouds.append(partial_to_add.numpy())
            else:
                partial_clouds.append(dss.numpy())
                partial_clouds.append(ess.numpy())

        for j in range(test_instances):
            partial_to_test = self.subsample_eye_side()
            tm, tn = partial_to_test.shape
            tm_new = tm
            if tm > len(partial_clouds[0]):
                tm_new = len(partial_clouds[0])
                torch.manual_seed(self.seed)
                rand_test_idx = torch.randperm(tm)
                test_subsample_indices = rand_test_idx[:tm_new]
                partial_to_test = partial_to_test[test_subsample_indices, :]
            else:
                tm_needed = len(partial_clouds[0]) - tm
                torch.manual_seed(self.seed)
                rand_needed_idx = torch.randperm(tm)
                needed_indices = rand_needed_idx[:tm_needed]
                partial_to_test = np.concatenate(partial_to_test, partial_to_test[needed_indices, :])
            partial_clouds_test.append(partial_to_test.numpy())
            label_choice = random.choice([1, 0])
            if label_choice == 1:
                labels_test.append(np.array([1, 0]))
            else:
                labels_test.append(np.array([0, 1]))

        return np.array(full_clouds_positive), np.array(full_clouds_negative), np.array(partial_clouds), np.array(
            labels), np.array(partial_clouds_test), np.array(labels_test)
