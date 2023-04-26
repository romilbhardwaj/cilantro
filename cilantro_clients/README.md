# Cilantro client design

A cilantro client is the remote unit of cilantro which resides with the application and sends UtilityUpdates to the CilantroScheduler over gRPC. It is expected to be running remotely, and thus is part of the separate `cilantro_clients` library.

A cilantro client consists of:
 1. `DataSource` which generates data with the `get_data()` method.
 2. `Publisher` which forwards the generated data to an appropriate sink. Sink examples include stdout, or a remote gRPC server (CilantroScheduler).
 
To extend the cilantro client, simply implement a new DataSource and pass it to the BaseCilantroClient. Good starting examples are `timeseries_to_stdout_driver.py` and `dummy_to_stdout_driver.py`. 