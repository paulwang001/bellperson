use std::sync::{Arc,Mutex};
use std::time::{Instant, Duration};

use crate::bls::Engine;
use ff::{Field, PrimeField};
use groupy::{CurveAffine, CurveProjective};
use rand_core::RngCore;
use rayon::prelude::*;

use super::{ParameterSource, Proof};
use crate::domain::{EvaluationDomain, Scalar};
use crate::gpu::{LockedFFTKernel, LockedMultiexpKernel};
use crate::multicore::{Worker, THREAD_POOL};
use crate::multiexp::{multiexp, DensityTracker, FullDensity};
use crate::{
    Circuit, ConstraintSystem, Index, LinearCombination, SynthesisError, Variable, BELLMAN_VERSION,
};
use log::{info,trace,error};

#[cfg(feature = "gpu")]
use crate::gpu::PriorityLock;
use lazy_static::{*};
use std::env;
use mempool::Semphore;



lazy_static!(

    pub static ref C2_CPU_TASKS:Semphore = create_cpu_lock();
    pub static ref C2_CPU_CIRCUIT:Semphore = create_circuit_lock();
    pub static ref C2_GPU_LOCK: Arc<Mutex<()>> = Arc::new(Mutex::new(()));

);
fn create_cpu_lock() ->Semphore{
    let cocurrent_tasks = if let Ok(val) = env::var("FIL_PROOFS_C2_TASKS") {
        if let Ok(val) = val.parse::<usize>() {
            val
        }else{
            opencl::Device::all().len()
        }
    }else{
        opencl::Device::all().len()
    };
    Semphore::new("cpu_task",cocurrent_tasks)
}
fn create_circuit_lock() ->Semphore{
    Semphore::new("cpu_circuit",get_circuit_tasks())
}

fn get_circuit_tasks() ->usize {
    if let Ok(val) = env::var("FIL_PROOFS_C2_CIRCUIT") {
        if let Ok(val) = val.parse::<usize>() {
            val
        }else{
            opencl::Device::all().len() + 1
        }
    }else{
        opencl::Device::all().len() + 1
    }
}

fn eval<E: Engine>(
    lc: &LinearCombination<E>,
    mut input_density: Option<&mut DensityTracker>,
    mut aux_density: Option<&mut DensityTracker>,
    input_assignment: &[E::Fr],
    aux_assignment: &[E::Fr],
) -> E::Fr {
    let mut acc = E::Fr::zero();

    for (&index, &coeff) in lc.0.iter() {
        let mut tmp;

        match index {
            Variable(Index::Input(i)) => {
                tmp = input_assignment[i];
                if let Some(ref mut v) = input_density {
                    v.inc(i);
                }
            }
            Variable(Index::Aux(i)) => {
                tmp = aux_assignment[i];
                if let Some(ref mut v) = aux_density {
                    v.inc(i);
                }
            }
        }

        if coeff == E::Fr::one() {
            acc.add_assign(&tmp);
        } else {
            tmp.mul_assign(&coeff);
            acc.add_assign(&tmp);
        }
    }

    acc
}

struct ProvingAssignment<E: Engine> {
    // Density of queries
    a_aux_density: DensityTracker,
    b_input_density: DensityTracker,
    b_aux_density: DensityTracker,

    // Evaluations of A, B, C polynomials
    a: Vec<Scalar<E>>,
    b: Vec<Scalar<E>>,
    c: Vec<Scalar<E>>,

    // Assignments of variables
    input_assignment: Vec<E::Fr>,
    aux_assignment: Vec<E::Fr>,
}
use std::fmt;
use std::sync::mpsc::sync_channel;
use rust_gpu_tools::opencl;

impl<E: Engine> fmt::Debug for ProvingAssignment<E> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("ProvingAssignment")
            .field("a_aux_density", &self.a_aux_density)
            .field("b_input_density", &self.b_input_density)
            .field("b_aux_density", &self.b_aux_density)
            .field(
                "a",
                &self
                    .a
                    .iter()
                    .map(|v| format!("Fr({:?})", v.0))
                    .collect::<Vec<_>>(),
            )
            .field(
                "b",
                &self
                    .b
                    .iter()
                    .map(|v| format!("Fr({:?})", v.0))
                    .collect::<Vec<_>>(),
            )
            .field(
                "c",
                &self
                    .c
                    .iter()
                    .map(|v| format!("Fr({:?})", v.0))
                    .collect::<Vec<_>>(),
            )
            .field("input_assignment", &self.input_assignment)
            .field("aux_assignment", &self.aux_assignment)
            .finish()
    }
}

impl<E: Engine> PartialEq for ProvingAssignment<E> {
    fn eq(&self, other: &ProvingAssignment<E>) -> bool {
        self.a_aux_density == other.a_aux_density
            && self.b_input_density == other.b_input_density
            && self.b_aux_density == other.b_aux_density
            && self.a == other.a
            && self.b == other.b
            && self.c == other.c
            && self.input_assignment == other.input_assignment
            && self.aux_assignment == other.aux_assignment
    }
}

impl<E: Engine> ConstraintSystem<E> for ProvingAssignment<E> {
    type Root = Self;

    fn new() -> Self {
        Self {
            a_aux_density: DensityTracker::new(),
            b_input_density: DensityTracker::new(),
            b_aux_density: DensityTracker::new(),
            a: vec![],
            b: vec![],
            c: vec![],
            input_assignment: vec![],
            aux_assignment: vec![],
        }
    }

    fn alloc<F, A, AR>(&mut self, _: A, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<E::Fr, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        self.aux_assignment.push(f()?);
        self.a_aux_density.add_element();
        self.b_aux_density.add_element();

        Ok(Variable(Index::Aux(self.aux_assignment.len() - 1)))
    }

    fn alloc_input<F, A, AR>(&mut self, _: A, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<E::Fr, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        self.input_assignment.push(f()?);
        self.b_input_density.add_element();

        Ok(Variable(Index::Input(self.input_assignment.len() - 1)))
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, _: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<E>) -> LinearCombination<E>,
        LB: FnOnce(LinearCombination<E>) -> LinearCombination<E>,
        LC: FnOnce(LinearCombination<E>) -> LinearCombination<E>,
    {
        let a = a(LinearCombination::zero());
        let b = b(LinearCombination::zero());
        let c = c(LinearCombination::zero());

        self.a.push(Scalar(eval(
            &a,
            // Inputs have full density in the A query
            // because there are constraints of the
            // form x * 0 = 0 for each input.
            None,
            Some(&mut self.a_aux_density),
            &self.input_assignment,
            &self.aux_assignment,
        )));
        self.b.push(Scalar(eval(
            &b,
            Some(&mut self.b_input_density),
            Some(&mut self.b_aux_density),
            &self.input_assignment,
            &self.aux_assignment,
        )));
        self.c.push(Scalar(eval(
            &c,
            // There is no C polynomial query,
            // though there is an (beta)A + (alpha)B + C
            // query for all aux variables.
            // However, that query has full density.
            None,
            None,
            &self.input_assignment,
            &self.aux_assignment,
        )));
    }

    fn push_namespace<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn pop_namespace(&mut self) {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }

    fn is_extensible() -> bool {
        true
    }

    fn extend(&mut self, other: Self) {
        self.a_aux_density.extend(other.a_aux_density, false);
        self.b_input_density.extend(other.b_input_density, true);
        self.b_aux_density.extend(other.b_aux_density, false);

        self.a.extend(other.a);
        self.b.extend(other.b);
        self.c.extend(other.c);

        self.input_assignment
            // Skip first input, which must have been a temporarily allocated one variable.
            .extend(&other.input_assignment[1..]);
        self.aux_assignment.extend(other.aux_assignment);
    }
}

pub fn create_random_proof_batch_priority<E, C, R, P: ParameterSource<E>>(
    circuits: Vec<C>,
    params: P,
    rng: &mut R,
    priority: bool,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: Engine,
    C: Circuit<E> + Send,
    R: RngCore,
{
    let r_s = (0..circuits.len()).map(|_| E::Fr::random(rng)).collect();
    let s_s = (0..circuits.len()).map(|_| E::Fr::random(rng)).collect();

    // let r_s = (0..circuits.len()).map(|_| E::Fr::one()).collect();
    // let s_s = (0..circuits.len()).map(|_| E::Fr::zero()).collect();

    create_proof_batch_priority::<E, C, P>(circuits, params, r_s, s_s, priority)
}

pub fn create_proof_batch_priority<E, C, P: ParameterSource<E>>(
    circuits: Vec<C>,
    params: P,
    r_s: Vec<E::Fr>,
    s_s: Vec<E::Fr>,
    priority: bool,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: Engine,
    C: Circuit<E> + Send,
{
    info!("Bellperson {} is being used!", BELLMAN_VERSION);
    let mode_fifo = crate::gpu::prove_mode();
    if mode_fifo.to_uppercase() != "N"  {
        if mode_fifo.to_uppercase() != "F" {
            log::info!("channel mode");
            THREAD_POOL.install(|| create_proof_batch_priority_channel(circuits, params, r_s, s_s, priority))
        }
        else{
            log::info!("fifo mode");
            THREAD_POOL.install(|| create_proof_batch_priority_fifo(circuits, params, r_s, s_s, priority))
        }

    }
    else{
        log::info!("origin mode");
        THREAD_POOL.install(|| create_proof_batch_priority_inner(circuits, params, r_s, s_s, priority))
    }

}

fn create_proof_batch_priority_inner<E, C, P: ParameterSource<E>>(
    circuits: Vec<C>,
    params: P,
    r_s: Vec<E::Fr>,
    s_s: Vec<E::Fr>,
    priority: bool,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: Engine,
    C: Circuit<E> + Send,
{
    let pool = crate::create_local_pool();
    pool.install(||{

        (*C2_CPU_TASKS).get();
        trace!("synthesize circuit start");
        let now = std::time::Instant::now();

        let mut provers = circuits
            .into_par_iter()
            .map(|circuit| -> Result<_, SynthesisError> {
                let mut prover = ProvingAssignment::new();

                prover.alloc_input(|| "", || Ok(E::Fr::one()))?;

                circuit.synthesize(&mut prover)?;

                for i in 0..prover.input_assignment.len() {
                    prover.enforce(|| "", |lc| lc + Variable(Index::Input(i)), |lc| lc, |lc| lc);
                }

                Ok(prover)
            })
            .collect::<Result<Vec<_>, _>>()?;



        let worker = Worker::new();
        let input_len = provers[0].input_assignment.len();
        let vk = params.get_vk(input_len)?;
        let n = provers[0].a.len();

        // Make sure all circuits have the same input len.
        for prover in &provers {
            assert_eq!(
                prover.a.len(),
                n,
                "only equaly sized circuits are supported"
            );
        }

        let mut log_d = 0;
        while (1 << log_d) < n {
            log_d += 1;
        }
        C2_CPU_TASKS.put();
        //drop(lock_of_cpu);
        trace!("synthesize circuit finish,{}s",now.elapsed().as_secs());
        trace!("^^^^^^^^^^^^^^^^getting lock");
        // Start fft/multiexp prover timer
        let start = Instant::now();
        info!("starting proof timer");

        #[cfg(feature = "gpu")]
        let prio_lock = if priority {
            Some(PriorityLock::lock())
        } else {
            None
        };

        let mut fft_kern = Some(LockedFFTKernel::<E>::new(log_d, priority));

        let a_s = provers
            .iter_mut()
            .map(|prover| {
                let mut a =
                    EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.a, Vec::new()))?;
                let mut b =
                    EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.b, Vec::new()))?;
                let mut c =
                    EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.c, Vec::new()))?;

                a.ifft(&worker, &mut fft_kern)?;
                a.coset_fft(&worker, &mut fft_kern)?;
                b.ifft(&worker, &mut fft_kern)?;
                b.coset_fft(&worker, &mut fft_kern)?;
                c.ifft(&worker, &mut fft_kern)?;
                c.coset_fft(&worker, &mut fft_kern)?;

                a.mul_assign(&worker, &b);
                drop(b);
                a.sub_assign(&worker, &c);
                drop(c);
                a.divide_by_z_on_coset(&worker);
                a.icoset_fft(&worker, &mut fft_kern)?;
                let mut a = a.into_coeffs();
                let a_len = a.len() - 1;
                a.truncate(a_len);

                Ok(Arc::new(
                    a.into_iter().map(|s| s.0.into_repr()).collect::<Vec<_>>(),
                ))
            })
            .collect::<Result<Vec<_>, SynthesisError>>()?;

        drop(fft_kern);
        let mut multiexp_kern = Some(LockedMultiexpKernel::<E>::new(log_d, priority));

        let h_s = a_s
            .into_iter()
            .map(|a| {
                let h = multiexp(
                    &worker,
                    params.get_h(a.len())?,
                    FullDensity,
                    a,
                    &mut multiexp_kern,
                );
                Ok(h)
            })
            .collect::<Result<Vec<_>, SynthesisError>>()?;

        let input_assignments = provers
            .par_iter_mut()
            .map(|prover| {
                let input_assignment = std::mem::replace(&mut prover.input_assignment, Vec::new());
                Arc::new(
                    input_assignment
                        .into_iter()
                        .map(|s| s.into_repr())
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();

        let aux_assignments = provers
            .par_iter_mut()
            .map(|prover| {
                let aux_assignment = std::mem::replace(&mut prover.aux_assignment, Vec::new());
                Arc::new(
                    aux_assignment
                        .into_iter()
                        .map(|s| s.into_repr())
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();

        let l_s = aux_assignments
            .iter()
            .map(|aux_assignment| {
                let l = multiexp(
                    &worker,
                    params.get_l(aux_assignment.len())?,
                    FullDensity,
                    aux_assignment.clone(),
                    &mut multiexp_kern,
                );
                Ok(l)
            })
            .collect::<Result<Vec<_>, SynthesisError>>()?;

        let inputs = provers
            .into_iter()
            .zip(input_assignments.iter())
            .zip(aux_assignments.iter())
            .map(|((prover, input_assignment), aux_assignment)| {
                let a_aux_density_total = prover.a_aux_density.get_total_density();

                let (a_inputs_source, a_aux_source) =
                    params.get_a(input_assignment.len(), a_aux_density_total)?;

                let a_inputs = multiexp(
                    &worker,
                    a_inputs_source,
                    FullDensity,
                    input_assignment.clone(),
                    &mut multiexp_kern,
                );

                let a_aux = multiexp(
                    &worker,
                    a_aux_source,
                    Arc::new(prover.a_aux_density),
                    aux_assignment.clone(),
                    &mut multiexp_kern,
                );

                let b_input_density = Arc::new(prover.b_input_density);
                let b_input_density_total = b_input_density.get_total_density();
                let b_aux_density = Arc::new(prover.b_aux_density);
                let b_aux_density_total = b_aux_density.get_total_density();

                let (b_g1_inputs_source, b_g1_aux_source) =
                    params.get_b_g1(b_input_density_total, b_aux_density_total)?;

                let b_g1_inputs = multiexp(
                    &worker,
                    b_g1_inputs_source,
                    b_input_density.clone(),
                    input_assignment.clone(),
                    &mut multiexp_kern,
                );

                let b_g1_aux = multiexp(
                    &worker,
                    b_g1_aux_source,
                    b_aux_density.clone(),
                    aux_assignment.clone(),
                    &mut multiexp_kern,
                );

                let (b_g2_inputs_source, b_g2_aux_source) =
                    params.get_b_g2(b_input_density_total, b_aux_density_total)?;

                let b_g2_inputs = multiexp(
                    &worker,
                    b_g2_inputs_source,
                    b_input_density,
                    input_assignment.clone(),
                    &mut multiexp_kern,
                );
                let b_g2_aux = multiexp(
                    &worker,
                    b_g2_aux_source,
                    b_aux_density,
                    aux_assignment.clone(),
                    &mut multiexp_kern,
                );

                Ok((
                    a_inputs,
                    a_aux,
                    b_g1_inputs,
                    b_g1_aux,
                    b_g2_inputs,
                    b_g2_aux,
                ))
            })
            .collect::<Result<Vec<_>, SynthesisError>>()?;

        drop(multiexp_kern);

        #[cfg(feature = "gpu")]
        drop(prio_lock);
        info!("^^^^^^^^^^^^^^^^^^exp finished,{}s", now.elapsed().as_secs());
        info!("^^^^^^^^^^^^^^^^^^assign values");
        let proofs = h_s
            .into_iter()
            .zip(l_s.into_iter())
            .zip(inputs.into_iter())
            .zip(r_s.into_iter())
            .zip(s_s.into_iter())
            .map(
                |(
                     (((h, l), (a_inputs, a_aux, b_g1_inputs, b_g1_aux, b_g2_inputs, b_g2_aux)), r),
                     s,
                 )| {
                    if vk.delta_g1.is_zero() || vk.delta_g2.is_zero() {
                        // If this element is zero, someone is trying to perform a
                        // subversion-CRS attack.
                        return Err(SynthesisError::UnexpectedIdentity);
                    }

                    let mut g_a = vk.delta_g1.mul(r);
                    g_a.add_assign_mixed(&vk.alpha_g1);
                    let mut g_b = vk.delta_g2.mul(s);
                    g_b.add_assign_mixed(&vk.beta_g2);
                    let mut g_c;
                    {
                        let mut rs = r;
                        rs.mul_assign(&s);

                        g_c = vk.delta_g1.mul(rs);
                        g_c.add_assign(&vk.alpha_g1.mul(s));
                        g_c.add_assign(&vk.beta_g1.mul(r));
                    }
                    let mut a_answer = a_inputs.wait()?;
                    a_answer.add_assign(&a_aux.wait()?);
                    g_a.add_assign(&a_answer);
                    a_answer.mul_assign(s);
                    g_c.add_assign(&a_answer);

                    let mut b1_answer = b_g1_inputs.wait()?;
                    b1_answer.add_assign(&b_g1_aux.wait()?);
                    let mut b2_answer = b_g2_inputs.wait()?;
                    b2_answer.add_assign(&b_g2_aux.wait()?);

                    g_b.add_assign(&b2_answer);
                    b1_answer.mul_assign(r);

                    g_c.add_assign(&b1_answer);
                    g_c.add_assign(&h.wait()?);
                    g_c.add_assign(&l.wait()?);
                    let p = Proof {
                        a: g_a.into_affine(),
                        b: g_b.into_affine(),
                        c: g_c.into_affine(),
                    };
                    let mut out = vec![];
                    p.write(&mut out).unwrap();
                    let out = hex::encode(out);
                    log::debug!("my proof:{}",out);
                    Ok(p)
                },
            )
            .collect::<Result<Vec<_>, SynthesisError>>()?;

        let proof_time = start.elapsed();
        info!("prover time: {:?}", proof_time);

        Ok(proofs)
    })
}


fn create_proof_batch_priority_fifo<E, C, P: ParameterSource<E>>(
    circuits: Vec<C>,
    params: P,
    r_s: Vec<E::Fr>,
    s_s: Vec<E::Fr>,
    priority: bool,
) -> Result<Vec<Proof<E>>, SynthesisError>
    where
        E: Engine,
        C: Circuit<E> + Send,
{

    let pool = crate::create_local_pool();
    pool.install(|| {
        info!("create_proof_batch_priority_fifo-------------------start...");
        let task_now = std::time::Instant::now();
        let fifo_id = std::time::UNIX_EPOCH.elapsed().unwrap().as_secs();
        let worker = Worker::new();
        let proofs =
            circuits
                .into_par_iter()
                .zip(r_s.into_par_iter())
                .zip(s_s.into_par_iter())
                .map(|((circuit, r), s)| {
                    let mut prover = ProvingAssignment::new();
                    {
                        let name = format!("FIFO-{}", fifo_id);
                        let mut cpu_count = 1;
                        if let Ok(c) = std::env::var("FIL_PROOFS_CPU_COUNT"){
                            cpu_count = c.parse().unwrap_or(1);
                        }
                        let _l = crate::gpu::GPULock::lock_count_default(name.as_str(), cpu_count);
                        info!("--------------------circuit synthesize[{}]--------------------", name);
                        let now = std::time::Instant::now();

                        prover.alloc_input(|| "", || Ok(E::Fr::one())).unwrap();

                        circuit.synthesize(&mut prover).unwrap();

                        for i in 0..prover.input_assignment.len() {
                            prover.enforce(|| "", |lc| lc + Variable(Index::Input(i)), |lc| lc, |lc| lc);
                        }
                        info!("--------------------circuit synthesized: {} s --------------------", now.elapsed().as_secs());
                    }
                    let lock = crate::gpu::GPULock::lock_count_default("CC", u8::MAX);
                    let mut n = 0;
                    let mut log_d = 0;
                    n = prover.a.len();
                    while (1 << log_d) < n {
                        log_d += 1;
                    }
                    info!("prover FFT:{}", n);

                    let aa = {
                        let mut a =
                            EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.a, Vec::new())).unwrap();

                        let mut b =
                            EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.b, Vec::new())).unwrap();

                        let mut c =
                            EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.c, Vec::new())).unwrap();
                        #[cfg(feature = "gpu")]
                        let prio_lock = if priority {
                            Some(PriorityLock::lock())
                        } else {
                            None
                        };
                        let mut fft_kern = Some(LockedFFTKernel::<E>::new(log_d, priority));
                        let now = std::time::Instant::now();
                        a.ifft(&worker, &mut fft_kern).unwrap();
                        a.coset_fft(&worker, &mut fft_kern).unwrap();

                        b.ifft(&worker, &mut fft_kern).unwrap();
                        b.coset_fft(&worker, &mut fft_kern).unwrap();

                        c.ifft(&worker, &mut fft_kern).unwrap();
                        c.coset_fft(&worker, &mut fft_kern).unwrap();

                        a.mul_assign(&worker, &b);
                        drop(b);
                        a.sub_assign(&worker, &c);
                        drop(c);
                        a.divide_by_z_on_coset(&worker);
                        {
                            a.icoset_fft(&worker, &mut fft_kern).unwrap();
                        }
                        let mut a = a.into_coeffs();
                        let a_len = a.len() - 1;
                        a.truncate(a_len);
                        drop(fft_kern);
                        #[cfg(feature = "gpu")]
                        drop(prio_lock);
                        info!("--------------------prover FFT finished, use: {} s --------------------", now.elapsed().as_secs());
                        Arc::new(
                            a.into_iter().map(|s| s.0.into_repr()).collect::<Vec<_>>(),
                        )
                    };

                    #[cfg(feature = "gpu")]
                    let prio_lock = if priority {
                        Some(PriorityLock::lock())
                    } else {
                        None
                    };
                    let mut multiexp_kern = Some(LockedMultiexpKernel::<E>::new(log_d, priority));
                    let now = std::time::Instant::now();
                    trace!("--------------------prover mutilexp start-------------------------");
                    let h = multiexp(
                        &worker,
                        params.get_h(aa.len()).expect("get h failed"),
                        FullDensity,
                        aa,
                        &mut multiexp_kern,
                    );

                    let input_assignment = std::mem::replace(&mut prover.input_assignment, Vec::new());
                    let a_input_assignment = Arc::new(
                        input_assignment
                            .into_iter()
                            .map(|s| s.into_repr())
                            .collect::<Vec<_>>(),
                    );

                    let aux_assignment = std::mem::replace(&mut prover.aux_assignment, Vec::new());
                    let a_aux_assignment = Arc::new(
                        aux_assignment
                            .into_iter()
                            .map(|s| s.into_repr())
                            .collect::<Vec<_>>()
                    );

                    let l = multiexp(
                        &worker,
                        params.get_l(a_aux_assignment.len()).unwrap(),
                        FullDensity,
                        a_aux_assignment.clone(),
                        &mut multiexp_kern,
                    );

                    let a_aux_density_total = prover.a_aux_density.get_total_density();

                    let (a_inputs_source, a_aux_source) =
                        params.get_a(a_input_assignment.len(), a_aux_density_total).unwrap();

                    let a_inputs = multiexp(
                        &worker,
                        a_inputs_source,
                        FullDensity,
                        a_input_assignment.clone(),
                        &mut multiexp_kern,
                    );

                    let input_len = prover.input_assignment.len();
                    let a_aux = multiexp(
                        &worker,
                        a_aux_source,
                        Arc::new(prover.a_aux_density),
                        a_aux_assignment.clone(),
                        &mut multiexp_kern,
                    );

                    let b_input_density = Arc::new(prover.b_input_density);
                    let b_input_density_total = b_input_density.get_total_density();
                    let b_aux_density = Arc::new(prover.b_aux_density);
                    let b_aux_density_total = b_aux_density.get_total_density();

                    let (b_g1_inputs_source, b_g1_aux_source) =
                        params.get_b_g1(b_input_density_total, b_aux_density_total).unwrap();

                    let b_g1_inputs = multiexp(
                        &worker,
                        b_g1_inputs_source,
                        b_input_density.clone(),
                        a_input_assignment.clone(),
                        &mut multiexp_kern,
                    );

                    let b_g1_aux = multiexp(
                        &worker,
                        b_g1_aux_source,
                        b_aux_density.clone(),
                        a_aux_assignment.clone(),
                        &mut multiexp_kern,
                    );

                    let (b_g2_inputs_source, b_g2_aux_source) =
                        params.get_b_g2(b_input_density_total, b_aux_density_total).unwrap();
                    let b_g2_inputs = multiexp(
                        &worker,
                        b_g2_inputs_source,
                        b_input_density,
                        a_input_assignment.clone(),
                        &mut multiexp_kern,
                    );
                    let b_g2_aux = multiexp(
                        &worker,
                        b_g2_aux_source,
                        b_aux_density,
                        a_aux_assignment.clone(),
                        &mut multiexp_kern,
                    );
                    drop(multiexp_kern);
                    #[cfg(feature = "gpu")]
                    drop(prio_lock);
                    info!("--------------------prover mutilexp finished. use:{} s --------------------", now.elapsed().as_secs());
                    // drop(a_input_assignment);
                    let vk = params.get_vk(input_len).unwrap();
                    if vk.delta_g1.is_zero() || vk.delta_g2.is_zero() {
                        // If this element is zero, someone is trying to perform a
                        // subversion-CRS attack.
                        return Err(SynthesisError::UnexpectedIdentity);
                    }
                    let mut g_a = vk.delta_g1.mul(r);
                    g_a.add_assign_mixed(&vk.alpha_g1);
                    let mut g_b = vk.delta_g2.mul(s);
                    g_b.add_assign_mixed(&vk.beta_g2);
                    let mut g_c;
                    {
                        let mut rs = r;
                        rs.mul_assign(&s);

                        g_c = vk.delta_g1.mul(rs);
                        g_c.add_assign(&vk.alpha_g1.mul(s));
                        g_c.add_assign(&vk.beta_g1.mul(r));
                    }
                    //   trace!("=========== a answer ....");
                    let mut a_answer = a_inputs.wait().unwrap();
                    a_answer.add_assign(&a_aux.wait().unwrap());
                    g_a.add_assign(&a_answer);
                    a_answer.mul_assign(s);
                    g_c.add_assign(&a_answer);

                    let mut b1_answer = b_g1_inputs.wait().unwrap();
                    b1_answer.add_assign(&b_g1_aux.wait().unwrap());
                    let mut b2_answer = b_g2_inputs.wait().unwrap();
                    b2_answer.add_assign(&b_g2_aux.wait().unwrap());

                    g_b.add_assign(&b2_answer);
                    b1_answer.mul_assign(r);

                    // trace!("=========== creating c ....");
                    g_c.add_assign(&b1_answer);
                    g_c.add_assign(&h.wait().unwrap());
                    g_c.add_assign(&l.wait().unwrap());

                    let p = Proof {
                        a: g_a.into_affine(),
                        b: g_b.into_affine(),
                        c: g_c.into_affine(),
                    };
                    drop(g_a);
                    drop(g_b);
                    drop(g_c);
                    let mut out = vec![];
                    p.write(&mut out).unwrap();
                    let out = hex::encode(out);
                    log::debug!("my proof fifo:{}", out);
                    drop(out);
                    drop(lock);
                    Ok(p)
                }).collect::<Result<Vec<_>, SynthesisError>>()?;
        trace!("~~~~~~~~~~~~~~~~proofs total time:{} min~~~~~~~~~~~~~~~~~", task_now.elapsed().as_secs_f32() / 60.0);
        Ok(proofs)
    })
    // rayon::scope_fifo(|sp|{
    //     info!("create_proof_batch_priority_fifo-------------------start...");
    //     let task_now = std::time::Instant::now();
    //     (*C2_CPU_TASKS).get();
    //     // let pool = ThreadPoolBuilder::new().create().unwrap();
    //
    //     info!("synthesize circuit start");
    //     info!("------------------gpu locked-------------");
    //
    //         drop(tx);
    //         let worker = Worker::new();
    //         let mut n = 0;
    //         let mut log_d = 0;
    //         let mut input_len = 0 ;
    //         let mut ovk = None;
    //         let mut ret = Ok(vec![]);
    //         let mut itr_rs = r_s.into_iter();
    //         let mut itr_ss = s_s.into_iter();
    //         let mut proofs = vec![];
    //
    //
    //     if ret.is_ok() && !proofs.is_empty() {
    //         ret = Ok(proofs);
    //     }
    //     trace!("~~~~~~~~~~~~~~~~proofs total time:{} min~~~~~~~~~~~~~~~~~",task_now.elapsed().as_secs_f32()/60.0);
    //     (*C2_CPU_TASKS).put();
    //     ret
    // })
}


fn create_proof_batch_priority_channel<E, C, P: ParameterSource<E>>(
    circuits: Vec<C>,
    params: P,
    r_s: Vec<E::Fr>,
    s_s: Vec<E::Fr>,
    priority: bool,
) -> Result<Vec<Proof<E>>, SynthesisError>
    where
        E: Engine,
        C: Circuit<E> + Send,
{
    let pool = crate::create_local_pool();

    pool.install(|| {
        info!("proof start channel-------------------");
        rayon::scope_fifo(|sp|{


            (*C2_CPU_TASKS).get();
            info!("task start");
            let task_now = std::time::Instant::now();
            let arc_params = Arc::new(params);

            let mut itr_rs = r_s.into_iter();
            let mut itr_ss = s_s.into_iter();
            // let is_first = Arc::new(Mutex::new(0));
            let (fft_tx,fft_rx) = sync_channel(get_circuit_tasks());
            let (input_tx,input_rx) = sync_channel(0);
            let (proof_tx,proof_rx) = sync_channel(0);
            let mut index = 0usize;
            let mut proofs = vec![None;circuits.len()];
            if get_circuit_tasks() > 1 {
                (*C2_CPU_CIRCUIT).get();
            }
            circuits
                .into_iter()
                .for_each(|circuit| {

                    let r = itr_rs.next().unwrap();
                    let s = itr_ss.next().unwrap();
                    let sender = fft_tx.clone();
                    let p = arc_params.clone();
                    index += 1;
                    sp.spawn_fifo(move |_|{

                        (*C2_CPU_CIRCUIT).get();
                        info!("--------------------circuit synthesize...--------------------");
                        let now = std::time::Instant::now();

                        let mut prover = ProvingAssignment::new();

                        prover.alloc_input(|| "", || Ok(E::Fr::one())).unwrap();

                        circuit.synthesize(&mut prover).unwrap();

                        for i in 0..prover.input_assignment.len() {
                            prover.enforce(|| "", |lc| lc + Variable(Index::Input(i)), |lc| lc, |lc| lc);
                        }
                        info!("--------------------circuit synthesized: {} s --------------------",now.elapsed().as_secs());

                        sender.send((prover,r,s,p,index)).unwrap();
                        (*C2_CPU_CIRCUIT).put();
                    });
                });
            drop(fft_tx);
            if get_circuit_tasks() > 1 {
                std::thread::sleep(Duration::from_secs(60));
                (*C2_CPU_CIRCUIT).put();
            }
            let a_worker = Arc::new(Worker::new());
            let worker = a_worker.clone();
            sp.spawn_fifo(move |fft|{

                while let Ok((mut prover,r,s,params,index)) = fft_rx.recv() {
                    let tx = input_tx.clone();
                    let worker = worker.clone();
                    fft.spawn_fifo(move|_|{
                        let n = prover.a.len();

                        let mut log_d = 0;
                        while (1 << log_d) < n{
                            log_d += 1;
                        }
                        let lock = crate::gpu::GPULock::lock_count_default("CC", u8::MAX);
                        let now = std::time::Instant::now();
                        trace!("--------------------prover FFT start, logd:{}--------------------",log_d);
                        let mut fft_kern = Some(LockedFFTKernel::<E>::new(log_d, priority));

                        let mut a =
                            EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.a, Vec::new())).unwrap();

                        let mut b =
                            EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.b, Vec::new())).unwrap();

                        let mut c =
                            EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.c, Vec::new())).unwrap();
                        a.ifft(&worker, &mut fft_kern).unwrap();
                        //   trace!("^^^^^^^^^^^^^^^^ coset fft1");
                        a.coset_fft(&worker, &mut fft_kern).unwrap();

                        b.ifft(&worker, &mut fft_kern).unwrap();
                        b.coset_fft(&worker, &mut fft_kern).unwrap();

                        c.ifft(&worker, &mut fft_kern).unwrap();
                        c.coset_fft(&worker, &mut fft_kern).unwrap();

                        a.mul_assign(&worker, &b);
                        drop(b);
                        // trace!("^^^^^^^^^^^^^^^^ assign data c");
                        a.sub_assign(&worker, &c);
                        drop(c);
                        a.divide_by_z_on_coset(&worker);
                        a.icoset_fft(&worker, &mut fft_kern).unwrap();
                        let mut a = a.into_coeffs();
                        let a_len = a.len() - 1;
                        a.truncate(a_len);

                        let a_s = Arc::new(
                            a.into_iter().map(|s| s.0.into_repr()).collect::<Vec<_>>(),
                        );
                        drop(fft_kern);
                        drop(lock);
                        info!("--------------------prover FFT finished, use: {} s --------------------",now.elapsed().as_secs());
                        tx.send((prover,a_s,r,s,params,priority,index)).unwrap();
                    });
                }

            });

            let worker = a_worker.clone();
            //input for exp
            sp.spawn_fifo(move |exp|{

                while let Ok((
                                 mut prover,
                                 a_s,
                                 r,
                                 s,
                                 params,
                                 priority,
                                 index,
                             )) = input_rx.recv() {
                    let worker = worker.clone();
                    let tx = proof_tx.clone();
                    exp.spawn_fifo(move|_|{
                        let mut log_d = 0;
                        let n = prover.a.len();
                        while (1 << log_d) < n{
                            log_d += 1;
                        }
                        let lock = crate::gpu::GPULock::lock_count_default("CC", u8::MAX);
                        let now = std::time::Instant::now();
                        let mut multiexp_kern = Some(LockedMultiexpKernel::<E>::new(log_d, priority));
                        trace!("--------------------prover mutilexp start-------------------------");
                        let h = multiexp(
                            &worker,
                            params.get_h(a_s.len()).unwrap(),
                            FullDensity,
                            a_s,
                            &mut multiexp_kern,
                        );


                        let input_assignment = std::mem::replace(&mut prover.input_assignment, Vec::new());
                        let a_input_assignment = Arc::new(
                            input_assignment
                                .into_iter()
                                .map(|s| s.into_repr())
                                .collect::<Vec<_>>(),
                        );
                        let aux_assignment = std::mem::replace(&mut prover.aux_assignment, Vec::new());
                        let a_aux_assignment= Arc::new(
                            aux_assignment
                                .into_iter()
                                .map(|s| s.into_repr())
                                .collect::<Vec<_>>()
                        );

                        let l = multiexp(
                            &worker,
                            params.get_l(a_aux_assignment.len()).unwrap(),
                            FullDensity,
                            a_aux_assignment.clone(),
                            &mut multiexp_kern,
                        );

                        let a_aux_density_total = prover.a_aux_density.get_total_density();

                        let (a_inputs_source, a_aux_source) =
                            params.get_a(a_input_assignment.len(), a_aux_density_total).unwrap();

                        let a_inputs = multiexp(
                            &worker,
                            a_inputs_source,
                            FullDensity,
                            a_input_assignment.clone(),
                            &mut multiexp_kern,
                        );
                        let a_aux = multiexp(
                            &worker,
                            a_aux_source,
                            Arc::new(prover.a_aux_density),
                            a_aux_assignment.clone(),
                            &mut multiexp_kern,
                        );
                        let b_input_density = Arc::new(prover.b_input_density);
                        let b_input_density_total = b_input_density.get_total_density();
                        let b_aux_density = Arc::new(prover.b_aux_density);
                        let b_aux_density_total = b_aux_density.get_total_density();

                        let (b_g1_inputs_source, b_g1_aux_source) =
                            params.get_b_g1(b_input_density_total, b_aux_density_total).unwrap();
                        let b_g1_inputs = multiexp(
                            &worker,
                            b_g1_inputs_source,
                            b_input_density.clone(),
                            a_input_assignment.clone(),
                            &mut multiexp_kern,
                        );
                        let b_g1_aux = multiexp(
                            &worker,
                            b_g1_aux_source,
                            b_aux_density.clone(),
                            a_aux_assignment.clone(),
                            &mut multiexp_kern,
                        );

                        let (b_g2_inputs_source, b_g2_aux_source) =
                            params.get_b_g2(b_input_density_total, b_aux_density_total).unwrap();
                        let b_g2_inputs = multiexp(
                            &worker,
                            b_g2_inputs_source,
                            b_input_density,
                            a_input_assignment.clone(),
                            &mut multiexp_kern,
                        );
                        let b_g2_aux = multiexp(
                            &worker,
                            b_g2_aux_source,
                            b_aux_density,
                            a_aux_assignment.clone(),
                            &mut multiexp_kern,
                        );
                        drop(multiexp_kern);
                        drop(lock);
                        info!("--------------------prover mutilexp finished. use:{} s --------------------",now.elapsed().as_secs());
                        tx.send((
                            h,
                            l,
                            a_inputs,
                            a_aux,
                            b_g1_inputs,
                            b_g1_aux,
                            b_g2_inputs,
                            b_g2_aux,
                            r,
                            s,
                            params,
                            prover.input_assignment.len(),
                            index,
                        )).unwrap();
                    });
                }
            });

            while let Ok((
                             h,
                             l,
                             a_inputs,
                             a_aux,
                             b_g1_inputs,
                             b_g1_aux,
                             b_g2_inputs,
                             b_g2_aux,
                             r,
                             s,
                             params,
                             input_assignment_len,
                             index,
                         )) = proof_rx.recv() {
                let vk = params.get_vk(input_assignment_len).unwrap();
                let mut g_a = vk.delta_g1.mul(r);
                g_a.add_assign_mixed(&vk.alpha_g1);
                let mut g_b = vk.delta_g2.mul(s);
                g_b.add_assign_mixed(&vk.beta_g2);
                let mut g_c;
                {
                    let mut rs = r;
                    rs.mul_assign(&s);

                    g_c = vk.delta_g1.mul(rs);
                    g_c.add_assign(&vk.alpha_g1.mul(s));
                    g_c.add_assign(&vk.beta_g1.mul(r));
                }
                //   trace!("=========== a answer ....");
                let mut a_answer = a_inputs.wait().unwrap();
                a_answer.add_assign(&a_aux.wait().unwrap());
                g_a.add_assign(&a_answer);
                a_answer.mul_assign(s);
                g_c.add_assign(&a_answer);
                //  trace!("=========== creating b answer ....");
                let mut b1_answer = b_g1_inputs.wait().unwrap();
                b1_answer.add_assign(&b_g1_aux.wait().unwrap());
                let mut b2_answer = b_g2_inputs.wait().unwrap();
                b2_answer.add_assign(&b_g2_aux.wait().unwrap());

                g_b.add_assign(&b2_answer);
                b1_answer.mul_assign(r);

                // trace!("=========== creating c ....");
                g_c.add_assign(&b1_answer);
                g_c.add_assign(&h.wait().unwrap());
                g_c.add_assign(&l.wait().unwrap());
                trace!("===========a proof finished index:{} ===========",index);
                proofs[index-1]=Some(Proof::<E> {
                    a: g_a.into_affine(),
                    b: g_b.into_affine(),
                    c: g_c.into_affine(),
                });

            }
            (*C2_CPU_TASKS).put();
            let proofs = proofs.into_iter().map(|p|Ok(p.clone().unwrap())).collect::<Result<Vec<_>, SynthesisError>>()?;
            trace!("~~~~~~~~~~~~~~~~proofs,total time:{} min~~~~~~~~~~~~~~~~~",task_now.elapsed().as_secs_f32()/60.0);
            Ok(proofs)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::bls::{Bls12, Fr};
    use rand::Rng;
    use rand_core::SeedableRng;
    use rand_xorshift::XorShiftRng;

    #[test]
    fn test_proving_assignment_extend() {
        let mut rng = XorShiftRng::from_seed([
            0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
            0xbc, 0xe5,
        ]);

        for k in &[2, 4, 8] {
            for j in &[10, 20, 50] {
                let count: usize = k * j;

                let mut full_assignment = ProvingAssignment::<Bls12>::new();
                full_assignment
                    .alloc_input(|| "one", || Ok(Fr::one()))
                    .unwrap();

                let mut partial_assignments = Vec::with_capacity(count / k);
                for i in 0..count {
                    if i % k == 0 {
                        let mut p = ProvingAssignment::new();
                        p.alloc_input(|| "one", || Ok(Fr::one())).unwrap();
                        partial_assignments.push(p)
                    }

                    let index: usize = i / k;
                    let partial_assignment = &mut partial_assignments[index];

                    if rng.gen() {
                        let el = Fr::random(&mut rng);
                        full_assignment
                            .alloc(|| format!("alloc:{},{}", i, k), || Ok(el.clone()))
                            .unwrap();
                        partial_assignment
                            .alloc(|| format!("alloc:{},{}", i, k), || Ok(el))
                            .unwrap();
                    }

                    if rng.gen() {
                        let el = Fr::random(&mut rng);
                        full_assignment
                            .alloc_input(|| format!("alloc_input:{},{}", i, k), || Ok(el.clone()))
                            .unwrap();
                        partial_assignment
                            .alloc_input(|| format!("alloc_input:{},{}", i, k), || Ok(el))
                            .unwrap();
                    }

                    // TODO: LinearCombination
                }

                let mut combined = ProvingAssignment::new();
                combined.alloc_input(|| "one", || Ok(Fr::one())).unwrap();

                for assignment in partial_assignments.into_iter() {
                    combined.extend(assignment);
                }
                assert_eq!(combined, full_assignment);
            }
        }
    }
}
