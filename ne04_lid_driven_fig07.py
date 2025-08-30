from _plottools import prt_flow_2d, gridfig, prt_mask_2d, save, show
from d2n5_lid_driven import d2n5_lid_driven

nu_list = [0.005, 0.002, 0.001]

test1 = d2n5_lid_driven(h = 1/64, nu = nu_list[0])
test2 = d2n5_lid_driven(h = 1/64, nu = nu_list[1])
test3 = d2n5_lid_driven(h = 1/64, nu = nu_list[2])

test1.until_time(20)
test2.until_time(20)
test3.until_time(20)

fig, axs = gridfig(1, 3)
kwargs = dict(broken_streamlines = False, linewidth = 0.3, density = (1, 0.5))
u, v = test1.get_numerical_speed()
prt_flow_2d(test1.x, test1.y, u, v, fig=fig, ax=axs[0], xlabel=f"$\\nu = {test1.nu}$", cmap="copper", **kwargs)
u, v = test2.get_numerical_speed()
prt_flow_2d(test2.x, test2.y, u, v, fig=fig, ax=axs[1], xlabel=f"$\\nu = {test2.nu}$", cmap="copper", **kwargs)
u, v = test3.get_numerical_speed()
prt_flow_2d(test3.x, test3.y, u, v, fig=fig, ax=axs[2], xlabel=f"$\\nu = {test3.nu}$", cmap="copper", **kwargs)
fig.tight_layout()
save.prefix = "LidDriven"
save(fig)
show(fig, False)