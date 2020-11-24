use fs2::FileExt;
use log::{debug, info, warn};
use std::fs::File;
use std::path::PathBuf;
use std::sync::{Arc,Mutex};
use rust_gpu_tools::*;
use lazy_static::lazy_static;

const GPU_LOCK_NAME: &str = "bellman.gpu.lock";
const PRIORITY_LOCK_NAME: &str = "bellman.priority.lock";

#[derive(Debug,Copy,Clone)]
pub struct LockDevice(u32,u8);

lazy_static! {
    pub static ref LOCKABLE_DEVICES: Arc<Mutex<Vec<LockDevice>>> = {
        Arc::new(Mutex::new(all_bus()))
    };
}

fn tmp_path(filename: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(filename);
    p
}

pub fn all_bus() -> Vec<LockDevice> {
    let v_d = opencl::Device::all();
    let mut bus_ids = vec![];
    for v in v_d {
        let bus_id = v.bus_id();
        if bus_id.is_some() {
            bus_ids.push(LockDevice(bus_id.unwrap(),0));
        }
    }
    bus_ids
}


/// get one unlocked device and lock
pub fn get_one_device_and_lock(retry:u32) ->Option<opencl::Device>{

    let bus_id = {
        let lockable_device = LOCKABLE_DEVICES.clone();
        let mut lockable_device = lockable_device.lock().unwrap();
        let unlocked =
            lockable_device.iter().filter(|x|x.1 == 0).map(|x|x.0).collect::<Vec<u32>>();
        if unlocked.is_empty() && retry > 0{
            if retry % 30 == 0 {
                log::warn!("get one GPU retry.{}",retry);
            }
            else{
                log::trace!("get one GPU retry.{}",retry);
            }
            drop(lockable_device);
            std::thread::sleep(Duration::from_secs(3));
            return get_one_device_and_lock(retry -1);
        }
        if unlocked.is_empty() {
            return None;
        }

        let mut rng = rand::thread_rng();
        let idx = rng.gen_range(0,unlocked.len());
        let id = unlocked[idx];
        for (i,x) in lockable_device.iter().enumerate() {
           if x.0 == id {
               lockable_device[i] = LockDevice(id,1);
               break;
           }
        }
        id
    };

    opencl::Device::all_iter().find_map(|x| {
        if x.bus_id().unwrap() == bus_id {
            Some(x.clone())
        }
        else{
            None
        }
    })
}

pub fn get_all_device_and_lock(retry:u32)->Vec<u32> {

    let unlocked = {
        let lockable_device = LOCKABLE_DEVICES.clone();
        let mut lockable_device = lockable_device.lock().unwrap();
        lockable_device.iter().filter(|x|x.1 == 0).map(|x|x.0).collect::<Vec<u32>>()
    };
    let mut v_idx = vec![];
    if !unlocked.is_empty() {
        let lockable_device = LOCKABLE_DEVICES.clone();
        let mut lockable = lockable_device.lock().unwrap();
        for (i,x) in lockable.iter().enumerate() {
            if unlocked.contains(&x.0) {
                v_idx.push((x.0,i));
            }
        }
    }
    else if retry > 0{
        if retry % 30 == 0 {
            log::warn!("get gpu retry.{}",retry);
        }
        else{
            log::trace!("get gpu retry.{}",retry);
        }
        std::thread::sleep(Duration::from_secs(3));
        return get_all_device_and_lock(retry-1);
    }
    for (id,i) in v_idx {
        let lockable_device = LOCKABLE_DEVICES.clone();
        let mut lockable = lockable_device.lock().unwrap();
        lockable[i] = LockDevice(id,1);
    }
    unlocked
}


pub fn unlock_device(d: &opencl::Device) {
    unlock_bus_id(d.bus_id().unwrap());
}

pub fn unlock_bus_id(id: u32) {
    let lockable_device = LOCKABLE_DEVICES.clone();
    let mut lockable_device = lockable_device.lock().unwrap();
    for (i,x) in lockable_device.iter().enumerate() {
        if x.0 == id {
            lockable_device[i] = LockDevice(id,0);
            break;
        }
    }
}


/// `GPULock` prevents two kernel objects to be instantiated simultaneously.
#[derive(Debug)]
pub struct GPULock(File,u8);
impl GPULock {
    pub fn lock() -> GPULock {
        debug!("Acquiring GPU lock...");
        let f = File::create(tmp_path(GPU_LOCK_NAME)).unwrap();
        f.lock_exclusive().unwrap();
        debug!("GPU lock acquired!");
        GPULock(f,u8::MAX)
    }
    pub fn lock_custom(index:u8) -> GPULock {
        debug!("Acquiring GPU lock...I-{}",index);
        let f = File::create(tmp_path(format!("{}.{}",GPU_LOCK_NAME,index).as_str())).unwrap();
        f.lock_exclusive().unwrap();
        debug!("GPU lock acquired!I-{}",index);
        GPULock(f,index)
    }
}
impl Drop for GPULock {
    fn drop(&mut self) {
        self.0.unlock().expect("gpu unlock error");
        if self.1 < u8::MAX {
            unlock_bus_id(self.1 as u32);
        }
        debug!("GPU lock released!");
    }
}


/// `GPULock` prevents two kernel objects to be instantiated simultaneously.
#[derive(Debug)]
pub struct MultiGPULock(Vec<(File,u32)>);
impl MultiGPULock {
    pub fn lock(v_id:Vec<u32>) -> MultiGPULock {

        let mut v_file = vec![];
        for id in v_id {
            debug!("Acquiring GPU lock...bus_id:{}",id);
            let f = File::create(tmp_path(format!("{}.{}.dev",GPU_LOCK_NAME,id).as_str())).unwrap();
            f.lock_exclusive().unwrap();
            v_file.push((f,id));
            debug!("GPU lock acquired! bus_id:{}",id);
        }

        MultiGPULock(v_file)
    }

}
impl Drop for MultiGPULock {
    fn drop(&mut self) {
        let vv = &self.0;
        for (f,id) in vv {
            f.unlock().expect("GPU unlock error");
            unlock_bus_id(id.clone());
        }
        debug!("GPU lock released!");
    }
}


/// `PrioriyLock` is like a flag. When acquired, it means a high-priority process
/// needs to acquire the GPU really soon. Acquiring the `PriorityLock` is like
/// signaling all other processes to release their `GPULock`s.
/// Only one process can have the `PriorityLock` at a time.
#[derive(Debug)]
pub struct PriorityLock(File);
impl PriorityLock {
    pub fn lock() -> PriorityLock {
        debug!("Acquiring priority lock...");
        let f = File::create(tmp_path(PRIORITY_LOCK_NAME)).unwrap();
        f.lock_exclusive().unwrap();
        debug!("Priority lock acquired!");
        PriorityLock(f)
    }
    pub fn wait(priority: bool) {
        if !priority {
            File::create(tmp_path(PRIORITY_LOCK_NAME))
                .unwrap()
                .lock_exclusive()
                .unwrap();
        }
    }
    pub fn should_break(priority: bool) -> bool {
        !priority
            && File::create(tmp_path(PRIORITY_LOCK_NAME))
                .unwrap()
                .try_lock_exclusive()
                .is_err()
    }
}
impl Drop for PriorityLock {
    fn drop(&mut self) {
        debug!("Priority lock released!");
    }
}

use super::error::{GPUError, GPUResult};
use super::fft::FFTKernel;
use super::multiexp::MultiexpKernel;
use crate::bls::Engine;
use crate::domain::create_fft_kernel;
use crate::multiexp::create_multiexp_kernel;
use std::collections::HashMap;
use rand::{RngCore, Rng};
use std::time::Duration;

macro_rules! locked_kernel {
    ($class:ident, $kern:ident, $func:ident, $name:expr) => {
        pub struct $class<E>
        where
            E: Engine,
        {
            log_d: usize,
            priority: bool,
            kernel: Option<$kern<E>>,
        }

        impl<E> $class<E>
        where
            E: Engine,
        {
            pub fn new(log_d: usize, priority: bool) -> $class<E> {
                $class::<E> {
                    log_d,
                    priority,
                    kernel: None,
                }
            }

            fn init(&mut self) {
                if self.kernel.is_none() {
                    PriorityLock::wait(self.priority);
                    info!("GPU is available for {}!", $name);
                    self.kernel = $func::<E>(self.log_d, self.priority);
                }
            }

            fn free(&mut self) {
                if let Some(_kernel) = self.kernel.take() {
                    warn!(
                        "GPU acquired by a high priority process! Freeing up {} kernels...",
                        $name
                    );
                }
            }

            pub fn with<F, R>(&mut self, mut f: F) -> GPUResult<R>
            where
                F: FnMut(&mut $kern<E>) -> GPUResult<R>,
            {
                if std::env::var("BELLMAN_NO_GPU").is_ok() {
                    return Err(GPUError::GPUDisabled);
                }

                self.init();

                loop {
                    if let Some(ref mut k) = self.kernel {
                        match f(k) {
                            Err(GPUError::GPUTaken) => {
                                self.free();
                                self.init();
                            }
                            Err(e) => {
                                warn!("GPU {} failed! Falling back to CPU... Error: {}", $name, e);
                                return Err(e);
                            }
                            Ok(v) => return Ok(v),
                        }
                    } else {
                        return Err(GPUError::KernelUninitialized);
                    }
                }
            }
        }
    };
}

locked_kernel!(LockedFFTKernel, FFTKernel, create_fft_kernel, "FFT");
locked_kernel!(
    LockedMultiexpKernel,
    MultiexpKernel,
    create_multiexp_kernel,
    "Multiexp"
);
