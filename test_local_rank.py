import horovod.tensorflow as hvd
hvd.init()
print(hvd.local_rank())
