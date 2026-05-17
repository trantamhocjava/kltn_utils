from . import adacbm_algo, random_algo, submodular_algo


def get_select_concept_idx(
    select_concept_method,
    img_feat,
    concept_feat,
    concept2class,
    num_select_concepts,
    num_images_per_class,
    weight,
):
    if select_concept_method == "submodular":
        select_idx = submodular_algo.submodular_select(
            img_feat,
            concept_feat,
            concept2class,
            num_select_concepts,
            num_images_per_class,
            weight,
        )
    elif select_concept_method == "random":
        select_idx = random_algo.random_select(
            concept2class,
            num_select_concepts,
            num_images_per_class,
        )
    elif select_concept_method == "adacbm":
        select_idx = adacbm_algo.adacbm_selection(
            img_feat,
            concept_feat,
            concept2class,
            num_select_concepts,
            num_images_per_class,
            weight,
        )

    return select_idx
