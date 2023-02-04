// todo: These should be `const` in WGPU 0.15.

let H_BAR = 1.;
let M_ELEC = 1.;
let K_C = 1.;
let Q_PROT = 1.;
let Q_ELEC = 1.;

//let KE_COEFF = -2.0 * M_ELEC / (HBAR * HBAR);
let KE_COEFF = -2.; // todo: Can't get the proper one above working.

struct Cplx {
    real: f32,
    im: f32,
}

@group(0)
@binding(0)
var<storage, read> cplx_input: array<Cplx>;

@group(0)
@binding(1)
var<storage, read_write> cplx_output: array<f32>;


// Multiply a complex number.
fn mul_cplx(v1: Cplx, v2: Cplx) -> Cplx {
    return Cplx (
        v1.real * v2.real - v1.im * v2.im,
        v1.real * v2.im + v1.im * v2.real
    );
}

// Compute the norm squared of a complex number.
fn norm_sq(v: Cplx) -> f32 {
    var conj = Cplx ( v.real, -v.im );
    return mul_cplx(conj, v).real;
}

// VKguide.dev:
// "Most of the time people choose a workgroup size that is a multiple of 64,
// and not very big, as a bigger workgroup
// size has to reserve more memory, and could be split within multiple cores.
@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
//    let len = arrayLength(&cplx_input);
//    var len = 10; // todo temp
    let i = global_invocation_id.x;

    // i / 2 since we're returning a f32 array while using complex input?
    cplx_output[i] = norm_sq(cplx_input[i]);
}
