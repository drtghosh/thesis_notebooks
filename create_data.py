import random
import torch
import numpy as np
from scipy.spatial import distance


class DumbCirc:
    def __init__(self, circle_center_x, rec_long_side, rec_short_side):
        self.circle_center_x = circle_center_x
        self.rec_long_side = rec_long_side
        self.rec_short_side = rec_short_side
        self.circle_radius = distance.euclidean([circle_center_x, 0], [rec_long_side / 2, rec_short_side / 2])

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

        rand_dumbbell_idx = torch.randperm(dm)
        rand_eye_idx = torch.randperm(em)
        dumbbell_subsample_indices = rand_dumbbell_idx[:num_dumbbell_subsample]
        eye_subsample_indices = rand_eye_idx[:num_eye_subsample]

        return (dumbbell_samples, eye_samples,
                dumbbell_samples[dumbbell_subsample_indices, :], eye_samples[eye_subsample_indices, :])

    def create_dataset(self, instances=1000, ppu_choices=None, percent_choices=None, equal_size=False):
        full_clouds = []
        partial_clouds = []
        for i in range(instances):
            if ppu_choices is not None:
                ppu = random.choice(ppu_choices)
            else:
                ppu = random.choice([10, 25, 50, 100, 150, 200])
            if percent_choices is not None:
                percent = random.choice(percent_choices)
            else:
                percent = random.choice([0.05, 0.1, 0.15, 0.2, 0.25])
            ds, es, dss, ess = self.subsample_shape(ppu, percent, equal_size)
            full_clouds.append(ds.numpy())
            partial_clouds.append(dss.numpy())
            full_clouds.append(es.numpy())
            partial_clouds.append(ess.numpy())

        return full_clouds, partial_clouds