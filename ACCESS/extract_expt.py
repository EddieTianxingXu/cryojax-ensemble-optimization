from simulate import get_expt, _parse_args
import logging

if __name__ == "__main__":
    args = _parse_args()
    log_level = logging.DEBUG if args.debug else None
    logging.basicConfig(level=log_level)
    get_expt(
        star_path=args.star_path,
        loads_envelop=False,
        mrc_folder_path=args.mrc_folder_path,
        n_images_in_parallel=args.expt_batch_size,
        expt_dir=args.expt_dir,
    )