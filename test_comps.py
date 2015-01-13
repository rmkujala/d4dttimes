import numpy as np
import os
import unittest
import hashlib
from collections import Counter

import dataio
import comp_tower_tower_travel_times as ctimes


class Test(unittest.TestCase):

    def setUp(self):
        self.mob_str1 = (
            "0,2012-01-07 17:20:00,1454\n" +
            "0,2012-01-07 17:30:00,1454\n" +
            "1,2012-01-08 10:40:00,323\n" +
            "1,2012-01-08 19:00:00,323\n" +
            "1,2012-01-08 21:00:00,132\n" +
            "0,2012-01-07 18:40:00,1327\n" +
            "0,2012-01-07 18:50:00,132\n" +
            "1,2012-01-07 13:10:00,1461\n" +
            "1,2012-01-07 20:30:00,323\n" +
            "1,2012-01-09 11:00:00,132\n"
        )
        self.n_regions = 1500
        # just some number larger than any site id in the mob_str1

        self.mob_str2 = (
            "0,2012-01-07 17:20:00,1\n" +
            "0,2012-01-07 17:30:00,2\n" +
            "0,2012-01-07 18:40:00,2\n" +
            "0,2012-01-07 18:50:00,3\n" +
            "0,2012-01-07 19:10:00,2\n" +
            "0,2012-01-07 20:30:00,1\n"
        )

        self.mob_str3 = (
            "0,2012-01-07 17:20:00,1\n" +
            "0,2012-01-07 17:30:00,2\n" +
            "0,2012-01-07 18:40:00,4\n" +
            "0,2012-01-07 18:50:00,2\n" +
            "0,2012-01-07 19:10:00,3\n" +
            "0,2012-01-07 20:30:00,1\n"
        )

        self.mob_str4 = (
            "0,2012-01-07 17:20:00,1\n" +
            "0,2012-01-07 17:30:00,2\n" +
            "0,2012-01-07 17:40:00,2\n" +
            "0,2012-01-07 18:40:00,4\n" +
            "0,2012-01-07 18:50:00,2\n" +
            "0,2012-01-07 20:10:00,4\n" +
            "0,2012-01-07 20:30:00,3\n"
        )

        self.mob_str5 = (
            "0,2014-01-07 17:20:00,1\n" +
            "0,2014-01-07 17:20:00,2\n" +
            "1,2014-01-07 17:40:00,1\n" +
            "1,2014-01-07 17:50:00,2\n" +
            "2,2014-01-07 18:50:00,1\n" +
            "2,2014-01-07 20:10:00,3\n" +
            "2,2014-01-07 20:30:00,2\n"
        )

        self.mob_str6 = (
            "0,2014-01-07 17:40:00,1\n" +
            "0,2014-01-07 17:50:00,3\n" +
            "1,2014-01-07 18:50:00,3\n" +
            "0,2014-01-07 17:20:00,3\n" +
            "0,2014-01-07 17:30:00,2\n" +
            "1,2014-01-07 20:10:00,3\n" +
            "1,2014-01-07 20:30:00,2\n"
        )

        self.mob_strs = [
            self.mob_str1,
            self.mob_str2,
            self.mob_str3,
            self.mob_str4,
            self.mob_str5,
            self.mob_str6
        ]

        self.mob_test_fnames = []

        for mob_str in self.mob_strs:
            mob_fname = "/tmp/" + hashlib.sha224(mob_str).hexdigest() + ".CSV"
            self.mob_test_fnames.append(mob_fname)
            with open(mob_fname, "w") as f:
                f.write(mob_str)

    def test_comp_mob_travel_times(self):
        cell_tower_ids = [[132], [323], [1454]]
        n_cell_ids = len(cell_tower_ids)
        ttsb = np.zeros((n_cell_ids, n_cell_ids), dtype=object)
        for row in ttsb:
            for i in range(len(row)):
                row[i] = Counter()
        ttsb[1, 0][120] += 1
        ttsb[2, 0][80] += 1

        ttimes = ctimes._get_travel_times_between_cell_tower_groups(
            self.mob_test_fnames[0], cell_tower_ids
        )
        self.assert_similarity_of_2d_np_counter_arrays(ttsb, ttimes)

        # trickier computations with the second set:
        cell_tower_ids2 = [[1], [2], [3]]
        n_cell_ids = len(cell_tower_ids)
        ttsb2 = np.zeros((n_cell_ids, n_cell_ids), dtype=object)
        for row in ttsb2:
            for i in range(len(row)):
                row[i] = Counter()

        ttsb2[0, 1][10] += 1
        ttsb2[0, 2][90] += 1
        ttsb2[1, 2][10] += 1
        ttsb2[2, 1][20] += 1
        ttsb2[2, 0][100] += 1
        ttsb2[1, 0][80] += 1
        print ttsb2

        ttimes2 = ctimes._get_travel_times_between_cell_tower_groups(
            self.mob_test_fnames[1], cell_tower_ids2
        )
        self.assert_similarity_of_2d_np_counter_arrays(ttsb2, ttimes2)

        cell_tower_ids3 = [[1, 2], [3]]
        n_cell_ids = len(cell_tower_ids3)
        ttsb3 = np.zeros((n_cell_ids, n_cell_ids), dtype=object)
        for row in ttsb3:
            for i in range(len(row)):
                row[i] = Counter()

        ttsb3[0, 1][10] += 1
        ttsb3[1, 0][20] += 1

        ttimes3 = ctimes._get_travel_times_between_cell_tower_groups(
            self.mob_test_fnames[1], cell_tower_ids3
        )
        self.assert_similarity_of_2d_np_counter_arrays(ttsb3, ttimes3)

        cell_tower_ids4 = [[1454, 132], [1327], [323]]
        n_cell_ids = len(cell_tower_ids4)
        ttsb4 = np.zeros((n_cell_ids, n_cell_ids), dtype=object)
        for row in ttsb4:
            for i in range(len(row)):
                row[i] = Counter()
        ttsb4[0, 1][70] += 1
        ttsb4[1, 0][10] += 1
        ttsb4[2, 0][120] += 1

        ttimes4 = ctimes._get_travel_times_between_cell_tower_groups(
            self.mob_test_fnames[0], cell_tower_ids4
        )
        self.assert_similarity_of_2d_np_counter_arrays(ttsb4, ttimes4)

        # test multiple csvs:
        n_cell_ids = len(cell_tower_ids4)
        ttsb5 = np.zeros((n_cell_ids, n_cell_ids), dtype=object)
        for row in ttsb5:
            for i in range(len(row)):
                row[i] = Counter()
        ttsb5[0, 1][70] += 2
        ttsb5[1, 0][10] += 2
        ttsb5[2, 0][120] += 2

        prefix = "/tmp/test_ttimes5"
        ctimes.comp_travel_times_between_cell_tower_groups(
            cell_tower_ids4,
            csv_fnames=[self.mob_test_fnames[0], self.mob_test_fnames[0]],
            n_cpus=2,
            fname_prefix=prefix
        )
        ttimes5 = np.load(prefix + ".npy")
        self.assert_similarity_of_2d_np_counter_arrays(ttsb5, ttimes5)
        os.remove(prefix + ".npy")
        os.remove(prefix + "_info.pkl")

    def test__get_filtered_travel_times_between_ctgroups(self):

        cell_tower_ids1 = [[1], [3, 4]]
        cell_tower_ids2 = [[2], [1]]
        # testing that same results are obtained without filtering:
        for ctids in [cell_tower_ids1, cell_tower_ids2]:
            r1 = ctimes._get_filtered_travel_times_between_ctgroups(
                self.mob_test_fnames[2], ctids)
            r2 = ctimes._get_travel_times_between_cell_tower_groups(
                self.mob_test_fnames[2], ctids)
            self.assert_similarity_of_2d_np_counter_arrays(r1, r2)

        # Testing the filtering of loops and stops:
        cell_tower_ids3 = [[1], [3]]
        n_cell_ids = len(cell_tower_ids3)

        r1 = ctimes._get_filtered_travel_times_between_ctgroups(
            self.mob_test_fnames[2],
            cell_tower_ids3,
            filter_loops_and_stops=True)
        tt_wf = np.zeros((n_cell_ids, n_cell_ids), dtype=object)
        for row in tt_wf:
            for i in range(len(row)):
                row[i] = Counter()
        tt_wf[0, 1][30] = 1
        tt_wf[1, 0][80] = 1
        self.assert_similarity_of_2d_np_counter_arrays(tt_wf, r1)

        # test loops etc. with self.mob_str4:
        cell_tower_ids4 = [[1], [3]]
        n_cell_ids = len(cell_tower_ids4)

        r1 = ctimes._get_filtered_travel_times_between_ctgroups(
            self.mob_test_fnames[3],
            cell_tower_ids4,
            filter_loops_and_stops=True)
        tt_wf = np.zeros((n_cell_ids, n_cell_ids), dtype=object)
        for row in tt_wf:
            for i in range(len(row)):
                row[i] = Counter()

        tt_wf[0, 1][10 + 60 + 20] += 1
        self.assert_similarity_of_2d_np_counter_arrays(tt_wf, r1)

        # test multiple users with self.mob_str5:
        cell_tower_ids5 = [[1], [2]]
        n_cell_ids = len(cell_tower_ids5)
        r1 = ctimes._get_filtered_travel_times_between_ctgroups(
            self.mob_test_fnames[4],
            cell_tower_ids5,
            filter_loops_and_stops=True
        )
        tt_wf = np.zeros((n_cell_ids, n_cell_ids), dtype=object)
        for row in tt_wf:
            for i in range(len(row)):
                row[i] = Counter()
        tt_wf[0, 1][0] += 1
        tt_wf[0, 1][10] += 1
        tt_wf[0, 1][100] += 1
        self.assert_similarity_of_2d_np_counter_arrays(tt_wf, r1)

    def test__get_filtered_travel_times_between_ctgroups2(self):
        # new test for changing user:

        # test multiple users with self.mob_str5:
        cell_tower_ids = [[1], [2], [3]]
        n_cell_ids = len(cell_tower_ids)
        r1 = ctimes._get_filtered_travel_times_between_ctgroups(
            self.mob_test_fnames[5],
            cell_tower_ids,
            filter_loops_and_stops=True
        )
        tt_wf = np.zeros((n_cell_ids, n_cell_ids), dtype=object)
        for row in tt_wf:
            for i in range(len(row)):
                row[i] = Counter()
        tt_wf[1, 0][10] += 1
        tt_wf[2, 1][10] += 1
        tt_wf[2, 1][20] += 1
        tt_wf[0, 2][10] += 1
        tt_wf[1, 2][20] += 1
        tt_wf[2, 0][20] += 1
        self.assert_similarity_of_2d_np_counter_arrays(tt_wf, r1)

    def test__get_filtered_travel_times_between_ctgroups3(self):
        # test loops etc. with self.mob_str4:
        cell_tower_ids4 = [[1], [3]]
        n_cell_ids = len(cell_tower_ids4)

        valid_start_interval = [17 + 21 / 60., 17 + 19 / 60.]
        r1 = ctimes._get_filtered_travel_times_between_ctgroups(
            self.mob_test_fnames[3],
            cell_tower_ids4,
            filter_loops_and_stops=True,
            valid_start_interval=valid_start_interval)
        tt_wf = np.zeros((n_cell_ids, n_cell_ids), dtype=object)
        for row in tt_wf:
            for i in range(len(row)):
                row[i] = Counter()
        self.assert_similarity_of_2d_np_counter_arrays(tt_wf, r1)

    def test___timestamp_within_interval(self):
        fname = self.mob_test_fnames[0]
        interval1 = [22, 11.0]
        interval2 = [22, 22]  # filter out everything
        interval3 = [21, 20]
        data = dataio.read_mobility_csv(fname)
        data.sort(["user_id", "timestamp"], inplace=True)
        data = ctimes._transform_time_stamps_to_minutes(data)

        within_interval1 = np.array(
            [False, False, False, False, False,
             False, True, False, False, True]
        )
        within_interval2 = np.array(
            [False] * 10
        )
        within_interval3 = np.array(
            [True, True, True, True, True,
             False, True, True, True, True]
        )

        r1 = ctimes._timestamp_within_interval(
            data['timestamp'].values, interval1)
        r2 = ctimes._timestamp_within_interval(
            data['timestamp'].values, interval2)
        r3 = ctimes._timestamp_within_interval(
            data['timestamp'].values, interval3)

        self.assert_similarity_of_1d_np_arrays(within_interval1, r1)
        self.assert_similarity_of_1d_np_arrays(within_interval2, r2)
        self.assert_similarity_of_1d_np_arrays(within_interval3, r3)

        timestamp = data.iloc[5]['timestamp']
        assert not ctimes._timestamp_within_interval(timestamp, interval1)
        assert not ctimes._timestamp_within_interval(timestamp, interval2)
        assert not ctimes._timestamp_within_interval(timestamp, interval3)

        timestamp = data.iloc[0]['timestamp']
        assert not ctimes._timestamp_within_interval(timestamp, interval1)
        assert not ctimes._timestamp_within_interval(timestamp, interval2)
        assert ctimes._timestamp_within_interval(timestamp, interval3)

    def test_travel_time_length_as_steps(self):
        # test loops etc. with self.mob_str4:
        cell_tower_ids4 = [[1], [3], [2]]
        n_cell_ids = len(cell_tower_ids4)

        r1 = ctimes._get_filtered_travel_times_between_ctgroups(
            self.mob_test_fnames[3],
            cell_tower_ids4,
            travel_time_as_steps=True)

        tt_wf = np.zeros((n_cell_ids, n_cell_ids), dtype=object)
        for row in tt_wf:
            for i in range(len(row)):
                row[i] = Counter()
        tt_wf[0, 1][6] += 1
        tt_wf[0, 2][1] += 1
        tt_wf[2, 1][2] += 1
        self.assert_similarity_of_2d_np_counter_arrays(tt_wf, r1)

    def test_travel_time_filter_transitions(self):
                # test loops etc. with self.mob_str4:
        cell_tower_ids4 = [[1], [3], [2]]
        n_cell_ids = len(cell_tower_ids4)

        r1 = ctimes._get_filtered_travel_times_between_ctgroups(
            self.mob_test_fnames[3],
            cell_tower_ids4,
            filter_transitions=True)

        tt_wf = np.zeros((n_cell_ids, n_cell_ids), dtype=object)
        for row in tt_wf:
            for i in range(len(row)):
                row[i] = Counter()
        tt_wf[0, 1][190] += 1
        tt_wf[2, 1][100] += 1
        self.assert_similarity_of_2d_np_counter_arrays(tt_wf, r1)

    def assert_similarity_of_1d_np_arrays(self, arr1, arr2):
        assert len(arr1) == len(arr2)
        assert (arr1 == arr2).all()

    def assert_similarity_of_2d_np_counter_arrays(self, arr1, arr2):
        """
        Assert similarity of 2d np arrays, whose elements are
        collections.Counter objects
        """
        assert arr1.shape == arr2.shape
        for i in range(len(arr1)):
            for j in range(len(arr1)):
                c1 = arr1[i, j]
                c2 = arr2[i, j]
                assert c1 == c2

    def tearDown(self):
        for fname in self.mob_test_fnames:
            if os.path.exists(fname):
                os.remove(fname)
