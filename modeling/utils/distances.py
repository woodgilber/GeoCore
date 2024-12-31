import h3


def is_near_validation(lat: float, lon: float, to_lat: list, to_lon: list, threshold: float) -> bool:
    """
    Inputs:
        lat - latitude of the point we are comparing to the test set
        lon - longitude of the point we are comparing to the test set
        to_lat - array of latitudes of the validation set
        to_lon - array of longitudes of the validation set
        threshold - buffer of the validation set
    """
    source = (lat, lon)
    return any([h3.point_dist(source, destination, unit="km") <= threshold for destination in zip(to_lat, to_lon)])
