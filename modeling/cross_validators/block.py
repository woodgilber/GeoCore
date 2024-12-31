import logging
from collections import Counter

import h3
import numpy as np
import pandas as pd
from keplergl import KeplerGl
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import haversine_distances

from modeling.cross_validators.base import BaseCV
from modeling.cross_validators.build import XVALIDATOR_REGISTRY
from modeling.utils.distances import is_near_validation

logger = logging.getLogger(__name__)


@XVALIDATOR_REGISTRY.register()
class BlockCV(BaseCV):
    """
    This is a custom method that creates 1/ create blocks based on distance (this distance should be estimated from
    the variagram = TODO for future us) 2/ merge blocks together based on distances and number of points in the cluster
    https://www.sciencedirect.com/science/article/pii/S1569843223001887
    """

    REQUIRED_COLUMNS = ["LATITUDE", "LONGITUDE"]
    REQUIRED_CONFIG_FIELDS = ["n_folds"]

    def compute_folds(self):
        """
        Computes the folds with a buffer.
        """
        n_folds = self.get_n_splits()

        # step 1: clustering
        X = np.array([self.df.LONGITUDE, self.df.LATITUDE]).transpose()
        X_radians = np.radians(X)

        # convert distance from km to angular
        angular_distance = self.config.clustering_distance / 6371
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric=haversine_distances,
            linkage="complete",
            distance_threshold=angular_distance,
        ).fit(X_radians)
        self.df["blocks"] = clustering.labels_

        n_blocks = len(set(clustering.labels_))
        logger.info(f"BlockCV created {n_blocks} blocks")
        if n_blocks <= n_folds:
            raise ValueError(
                "The agglomerative clustering steps created less blocks than wanted fold. Try again with a higher"
                " clustering distance"
            )

        # create a temporary dataframe containing the centroids of each blocks
        mean_locs = pd.DataFrame(
            {
                "lat": self.df.groupby("blocks")["LATITUDE"].transform("mean"),
                "long": self.df.groupby("blocks")["LONGITUDE"].transform("mean"),
            }
        )
        mean_locs["blocks"] = self.df["blocks"]
        cter = Counter(mean_locs.blocks)
        mean_locs["counter"] = mean_locs.blocks.apply(lambda k: cter[k])
        mean_locs.drop_duplicates(inplace=True)

        # for each block, calculate centroid and distance to other blocks -> put in matrix
        # this step takes a little longer.
        logger.info("Computing distance matrix. This may take a while.")
        tmp_df = mean_locs.copy()
        tmp_df.drop_duplicates(inplace=True)

        n_blocks = len(set(clustering.labels_))
        dist_mat = np.zeros((n_blocks, n_blocks))
        for i in range(n_blocks):
            row = tmp_df[tmp_df.blocks == i]
            centroid1 = [row.lat, row.long]
            for j in range(n_blocks):
                row = tmp_df[tmp_df.blocks == j]
                centroid2 = [row.lat, row.long]
                dist_mat[i, j] = h3.point_dist(centroid1, centroid2)

        # find one point on the convex hull (eg, max lat)
        starter_block = int(mean_locs.sort_values(by=["lat", "long"]).iloc[-1, :].blocks)
        # we grow each cluster until a certain cluster size
        cluster_size = self.df.shape[0] // n_folds
        # blocks that have been matched to a cluster index are stored here
        matched_blocks = []
        cluster_index = 0
        # set the cluster index in the mean_locs dataframe
        mean_locs["final_clusters"] = None

        # keep going as long as we have not matched everything. For the last cluster, we just dump everything left
        while len(matched_blocks) != n_blocks and cluster_index < n_folds - 1:
            matched_blocks.append(starter_block)
            mean_locs.loc[mean_locs.blocks == starter_block, "final_clusters"] = cluster_index
            current_size = mean_locs[mean_locs.blocks == starter_block].counter[0]

            # find closest neighbors - first position is the same index, hence idx = 1
            closest = np.argsort(dist_mat[starter_block, :])
            idx = 1
            while current_size < cluster_size:
                neighbor = closest[idx]
                # we first check if this cluster has been matched
                if neighbor not in matched_blocks:
                    added_size = mean_locs[mean_locs.blocks == neighbor].counter[0]
                    # this is a safety check to make sure that adding the next block won't grow the cluster too fat
                    # ie, more than 10% of the wanted size
                    if current_size + added_size > cluster_size * 1.1:
                        break
                    matched_blocks.append(neighbor)
                    mean_locs.loc[mean_locs.blocks == neighbor, "final_clusters"] = cluster_index
                    current_size += added_size
                idx += 1

            # assign starter block to the closest unassigned neighbor
            starter_block = [k for k in closest if k not in matched_blocks][0]
            cluster_index += 1

        # dump the rest in the last cluster
        for i in range(n_blocks):
            if i not in matched_blocks:
                mean_locs.loc[mean_locs.blocks == i, "final_clusters"] = n_folds - 1

        # set clusters in main df
        self.df["clusters"] = self.df.blocks.apply(lambda k: mean_locs[mean_locs.blocks == k].final_clusters[0])

        self.computed_folds = []
        for index in range(n_folds):
            flags = ["validation" if k == index else "train" for k in self.df["clusters"]]
            ret = pd.DataFrame({"flags": flags})

            validation_coords = self.df.iloc[np.where(ret["flags"] == "validation")[0], :].copy()
            validations_lats = validation_coords.LATITUDE.to_list()
            validation_lons = validation_coords.LONGITUDE.to_list()

            for i in range(0, len(flags)):
                if flags[i] == "train":
                    if is_near_validation(
                        lat=self.df.LATITUDE[i],
                        lon=self.df.LONGITUDE[i],
                        to_lat=validations_lats,
                        to_lon=validation_lons,
                        threshold=self.config.buffer,
                    ):
                        ret.loc[i, "flags"] = "buffer"

            self.computed_folds.append(
                {
                    "flags": ret["flags"],
                    "train_indices": np.where(ret["flags"] == "train")[0],
                    "validation_indices": np.where(ret["flags"] == "validation")[0],
                    "buffer_indices": np.where(ret["flags"] == "buffer")[0],
                    "n_drop": (ret["flags"] == "drop").sum(),
                    "n_train": (ret["flags"] == "train").sum(),
                    "n_valid": (ret["flags"] == "validation").sum(),
                }
            )

    def fold_assign(self):
        return

    def save_fold_plot(self, folder: str) -> None:
        """create a keplergl map"""
        tmp_data = self.df.copy()
        plot_data = {
            "folds": tmp_data[["LONGITUDE", "LATITUDE", "clusters"]],
            "blocks": tmp_data[["LONGITUDE", "LATITUDE", "blocks"]],
        }
        KeplerGl(height=400, data=plot_data).save_to_html(file_name=f"{folder}/cv_fold_map.html", center_map=True)

        return plot_data

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""
        return self.config["n_folds"]
