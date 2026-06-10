from simulate import get_sim, _parse_args

if __name__ == "__main__":
    args = _parse_args()
    get_sim(
        star_path=args.star_path,
        loads_envelop=False,
        pdb_path=args.pdb_path,
        sim_dir=args.sim_dir,
        data_sign='dark-on-light',
        atom_selection="all",
        noise_snr_range=[1e10, 1e10], #SNR is dynamically determined in David's ll calculation, here store massive SNR
        images_per_file=2000,
        batch_size=args.sim_batch_size
    )