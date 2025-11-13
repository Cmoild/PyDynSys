    .section .text
    .globl lu_chen
    .type lu_chen, @function
# def lu_chen(x, y, z, a, b, c, u):
#     return [
#         a * (y - x),
#         (1 - z) * x + c * y + u,
#         x * y - b * z
#     ]

# void f(float *in, float *c, float *out)
#             %rdi      %rsi        %rdx
lu_chen:
    movups (%rdi), %xmm0 # in[0] ... in[3]
    movups (%rsi), %xmm1 # c[0] ... c[3]

    pshufd $0x55, %xmm0, %xmm7
    movss %xmm7, %xmm2 # y -> xmm2

    subss %xmm0, %xmm2 # y - x# xmm2 = xmm2 - xmm0

    mulss %xmm1, %xmm2 # a * xmm2

    movss %xmm2, (%rdx) # out[0]

    movss .LC0(%rip), %xmm2 # xmm2 = 1
    pshufd $0xAA, %xmm0, %xmm7 # z -> xmm7

    subss %xmm7, %xmm2 # 1 - z
    mulss %xmm0, %xmm2 # xmm2 * x

    pshufd $0x55, %xmm0, %xmm3 # y -> xmm3
    pshufd $0xAA, %xmm1, %xmm7 # c -> xmm7

    mulss %xmm7, %xmm3 # xmm3 = c * y
    addss %xmm3, %xmm2 # xmm2 += xmm3

    pshufd $0xFF, %xmm1, %xmm7 # u -> xmm7
    addss %xmm7, %xmm2 # xmm2 += u

    movss %xmm2, 0x4(%rdx)

    pshufd $0x55, %xmm0, %xmm2 # y -> xmm2
    mulss %xmm0, %xmm2 # y * x

    pshufd $0x55, %xmm1, %xmm7 # b -> xmm7
    pshufd $0xAA, %xmm0, %xmm3 # z -> xmm3
    mulss %xmm7, %xmm3 # xmm3 *= b

    subss %xmm3, %xmm2

    movss %xmm2, 0x8(%rdx)

    ret
    .size lu_chen, .-lu_chen

# struct IntegratorCPU {
#     float step <- offset 0
#     float* points <- offset 8
#     float* C <- offset 16
#     size_t cur_step <- offset 24
#     integratedFunc func <- offset 32
# }
# extern "C" void integrator_cd_lu_chen_step(IntegratorCPU* integrator)
#                                                           %rdi
#    new_x = x + a*(y-x) * dt / 2.
#    new_y = y + ((1-z)*new_x + c*y + u) * dt / 2.
#    new_z = z + (new_x*new_y - b*z) * dt / 2.
#    new_z = (new_z + dt / 2. * new_x * new_y) / (1 + dt / 2. * b)
#    new_y = (new_y + dt / 2. * (1 - new_z) * new_x + u) / (1 - dt / 2. * c)
#    new_x = (new_x + dt / 2. * a * new_y) / (1 + dt / 2. * a)

    .globl cd_lu_chen_step
    .type cd_lu_chen_step, @function
cd_lu_chen_step:
    movss (%rdi), %xmm7
    divss .LC1(%rip), %xmm7 # dt / 2

    mov 0x8(%rdi), %rax # integrator->points
    mov 0x18(%rdi), %rcx # integrator->cur_step
    mov 0x10(%rdi), %rdx #integrator->C
    imul $3, %rcx # idx

    movss (%rax, %rcx, 4), %xmm1 # x
    movss 0x4(%rax, %rcx, 4), %xmm0 # y

    subss %xmm1, %xmm0 # y - x
    mulss (%rdx), %xmm0 # xmm0 *= a
    mulss %xmm7, %xmm0 # xmm0 *= dt / 2
    addss %xmm1, %xmm0 # xmm0 += x

    movss .LC0(%rip), %xmm1
    subss 0x8(%rax, %rcx, 4), %xmm1 # 1 - z
    mulss %xmm0, %xmm1 # xmm1 *= new_x
    movss 0x4(%rax, %rcx, 4), %xmm2 # y
    mulss 0x8(%rdx), %xmm2 # y * c
    addss %xmm2, %xmm1 # += c*y
    addss 0xc(%rdx), %xmm1 # xmm1 += u
    mulss %xmm7, %xmm1 # xmm1 *= dt / 2
    addss 0x4(%rax, %rcx, 4), %xmm1 # xmm1 += y

    movss %xmm0, %xmm2
    mulss %xmm1, %xmm2 # new_x * new_y
    movss 0x8(%rax, %rcx, 4), %xmm3 # z
    mulss 0x4(%rdx), %xmm3 # z * b
    subss %xmm3, %xmm2
    mulss %xmm7, %xmm2
    addss 0x8(%rax, %rcx, 4), %xmm2 # new_z

    movss %xmm0, %xmm3
    mulss %xmm1, %xmm3 # new_x * new_y
    mulss %xmm7, %xmm3 # *= dt / 2
    addss %xmm3, %xmm2
    movss 0x4(%rdx), %xmm4 # b
    mulss %xmm7, %xmm4 # b * dt / 2
    addss .LC0(%rip), %xmm4 # + 1
    divss %xmm4, %xmm2 # new_z

    movss .LC0(%rip), %xmm3
    subss %xmm2, %xmm3 # 1 - new_z
    mulss %xmm0, %xmm3 # * new_x
    mulss %xmm7, %xmm3 # * dt / 2
    addss %xmm3, %xmm1 # + new_y
    addss 0xc(%rdx), %xmm1 # + u
    movss 0x8(%rdx), %xmm3 # c
    mulss %xmm7, %xmm3
    movss .LC0(%rip), %xmm4
    subss %xmm3, %xmm4 # 1 - dt / 2 * c
    divss %xmm4, %xmm1 # new_y

    movss (%rdx), %xmm3 # a
    mulss %xmm7, %xmm3
    mulss %xmm1, %xmm3
    addss %xmm3, %xmm0 # new_x + dt / 2. * a * new_y
    mulss (%rdx), %xmm7
    addss .LC0(%rip), %xmm7
    divss %xmm7, %xmm0 # new_x

    mov 0x18(%rdi), %rcx # integrator->cur_step
    inc %rcx
    mov %rcx, 0x18(%rdi)
    imul $3, %rcx # next_idx
    movss %xmm0, (%rax, %rcx, 4)
    movss %xmm1, 0x4(%rax, %rcx, 4)
    movss %xmm2, 0x8(%rax, %rcx, 4)
    ret
    .size cd_lu_chen_step, .-cd_lu_chen_step


.section .rodata
.align 4
.LC0: .float 1.0
.LC1: .float 2.0
