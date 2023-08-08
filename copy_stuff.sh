for x in {0..49}
do
	cp forces_energy_mol1_frame$x.txt /home/ac127777/Documents/methylphosphate/cp2k_geom_opts/mp0mp0/frame$x/forces_energy_mol1_frame$x.txt
	cp forces_energy_mol2_frame$x.txt /home/ac127777/Documents/methylphosphate/cp2k_geom_opts/mp0mp0/frame$x/forces_energy_mol2_frame$x.txt
	cp mol1_frame$x.out /home/ac127777/Documents/methylphosphate/cp2k_geom_opts/mp0mp0/frame$x/mol1_frame$x.out
	cp mol2_frame$x.out /home/ac127777/Documents/methylphosphate/cp2k_geom_opts/mp0mp0/frame$x/mol2_frame$x.out
done