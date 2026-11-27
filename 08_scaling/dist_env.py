# Set the torch.distributed environment from Open MPI (mpirun) or torchrun, and pick this rank's GPU.
import os, torch, torch.distributed as dist
def init():
    e = os.environ
    if "OMPI_COMM_WORLD_RANK" in e:
        e["RANK"], e["WORLD_SIZE"] = e["OMPI_COMM_WORLD_RANK"], e["OMPI_COMM_WORLD_SIZE"]
        e["LOCAL_RANK"] = e["OMPI_COMM_WORLD_LOCAL_RANK"]
    e.setdefault("MASTER_ADDR", "localhost"); e.setdefault("MASTER_PORT", "29500")
    torch.cuda.set_device(int(e.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl", device_id=torch.device("cuda", int(e.get("LOCAL_RANK", 0))))
    return dist.get_rank(), dist.get_world_size()
