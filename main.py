#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import uvicorn
from app.utils.env_patch import apply_openmp_patch
from app.iface.server import VPEServer

apply_openmp_patch()
app = VPEServer().get_app()

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
