from data_logger import DataLogger
from pandas import DataFrame as df


def test_basic_logging():
    logger = DataLogger(4, r"cilantro_data.csv")
    logger.log_data(1, {"load": [10], "utility":[100]})
    assert logger.latest_timestamp == 1, "Latest timestamp should be 1, actually is " + str(logger.latest_timestamp)
    expected_data = df.from_dict({"timestamp": [1], "load": [10], "utility":[100]})
    assert df.equals(logger.current_data, expected_data), "Log_data not producing expected result"
    
    return

def test_split_data():
    file = open("cilantro_data.csv","r+")
    file.truncate(0)
    file.close()
    logger = DataLogger(4, r"cilantro_data.csv")
    logger.log_data(1, {"load": [10], "utility":[100]})
    logger.log_data(2, {"load": [20], "utility":[200]})
    logger.log_data(3, {"load": [30], "utility":[300]})
    logger.log_data(4, {"load": [40], "utility":[400]})
    logger.log_data(5, {"load": [50], "utility":[500]})
    assert logger.earliest_timestamp == 4, "Earliest timestamp should be 4"
    assert logger.latest_timestamp == 5, "Latest timestamp should be 5"
    return

def test_get_data():
    file = open("cilantro_data.csv","r+")
    file.truncate(0)
    file.close()
    logger = DataLogger(4, r"cilantro_data.csv")
    logger.log_data(1, {"load": [10], "utility":[100]})
    logger.log_data(2, {"load": [20], "utility":[200]})
    logger.log_data(3, {"load": [30], "utility":[300]})
    res = logger.get_data(1, 3)
    expected_result = df({"timestamp": [1, 2, 3], "load": [10, 20, 30], "utility": [100, 200, 300]})
    assert df.equals(res, expected_result), "Get_data not producing expected result (1)"

    logger.log_data(4, {"load": [40], "utility":[400]})
    logger.log_data(5, {"load": [50], "utility":[500]})
    assert logger.earliest_timestamp == 4, "Earliest timestamp should be 4"
    assert logger.latest_timestamp == 5, "Latest timestamp should be 5"
    
    res = logger.get_data(1, 4)
    expected_result = df({"timestamp": [1, 2, 3, 4], "load": [10, 20, 30, 40], "utility": [100, 200, 300, 400]})
    assert df.equals(res, expected_result), "Get_data not producing expected result (2)"

    return

if __name__ == "__main__":
    test_basic_logging()
    print("Basic data logging tests pass")
    test_split_data()
    print("Split data tests pass")
    test_get_data()
    print("Get data tests pass")
    print("All tests pass")