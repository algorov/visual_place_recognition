import os


def apply_openmp_patch():
    """
    Fix OpenMP duplicate library issue for FAISS + PyTorch compatibility.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
