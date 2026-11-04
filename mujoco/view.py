import mujoco

m = mujoco.MjModel.from_xml_path("mujoco/cartpole.xml")
print(m.nq, m.nv, m.nu)