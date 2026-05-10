from optparse import OptionParser

import yaml


def main(config):
    print("demo yaml")
    print("config.last_state")
    print(config.last_state)


if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option(
        "--last_state",
        type="str",
        dest="last_state",
    )

    cfg, args = parser.parse_args()

    cfg.last_state = yaml.safe_load(cfg.last_state)

    main(cfg)
