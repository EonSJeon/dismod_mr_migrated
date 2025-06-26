import pymc as pm
import numpy as np
import networkx as nx
import dismod_mr_pymc5

import time
import pymc as pm
import arviz as az
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

def asr(
    mr_model: "dismod_mr_pymc5.data.MRModel",
    pm_model: pm.Model,
    data_type: str,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    cores: int = 4,
    target_accept: float = 0.9,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], pm.backends.arviz.InferenceData]:
    """
    Fit one age‐specific rate model via PyMC v5.3 (NUTS + MAP initialization).

    Parameters
    ----------
    mr_model : MRModel
        `setup_model`을 거쳐 `mr_model.vars[data_type]`에 PyMC 변수들이 채워져 있어야 합니다.
        또한, `pm_model` 인자로 넘긴 것은 이미 `with pm.Model(): …` 블록 안에서
        age-specific rate를 정의한 PyMC Model 객체여야 합니다.
    pm_model : pm.Model
        `mr_model.vars[data_type]`에 대응하는 PyMC 모델. 반드시 `with pm.Model(): …` 로 감싸서 넘겨야 함.
    data_type : str
        'p', 'i', 'r', 'f' 등 age‐specific rate type
    draws, tune, chains, cores, target_accept, verbose
        PyMC NUTS 샘플링 옵션

    Returns
    -------
    map_estimate : dict
        `pm.find_MAP()`가 반환한 MAP point estimate (parameter 이름 → 최적값).
    idata : arviz.InferenceData
        `pm.sample()` 결과로 얻은 InferenceData 객체.
    """
    logger.info("==== asr enabled ====")
    if verbose:
        logger.info("==== verbose mode enabled ====")
        print("==== verbose mode enabled ====")

    # --- 1) 입력 확인 ---
    assert pm_model is not None, "asr는 pm_model이 설정된 후 호출되어야 합니다."

    # mr_model.vars[data_type] 가 반드시 존재해야 함
    if not hasattr(mr_model, "vars") or data_type not in mr_model.vars:
        raise AssertionError(
            f"mr_model.vars[{data_type!r}] 가 없습니다. 먼저 `setup_model(..., rate_type=data_type)` 를 호출하세요."
        )
    
    vars_dict = mr_model.vars[data_type]

    if verbose:
        logger.info(f"[asr] data_type={data_type!r} 에 대해 MAP + NUTS 샘플링 시작")


    # --- 2) MAP 초기화 ---
    with pm_model:
        if verbose:
            logger.info("  ▶ pm.find_MAP() 수행 중...")

        map_estimate = pm.find_MAP()
        if verbose:
            logger.info("    MAP 결과:")
            for name, val in map_estimate.items():
                try:
                    scalar = val.eval()  # Tensor → numpy
                except Exception:
                    scalar = val
                logger.info(f"      {name}: {scalar}")


    # --- 3) NUTS 샘플링 (posterior) ---
    #    샘플링 전후 시간을 기록해서 wall_time 계산
    t_start = time.time()
    with pm_model:
        if verbose:
            logger.info("  ▶ pm.sample() 수행 중 (NUTS)...")

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            start=map_estimate,
            target_accept=target_accept,
            progressbar=verbose,
            return_inferencedata=True,  # 반드시 InferenceData 형태로 반환
        )
        if verbose:
            logger.info("  ▶ 샘플링 완료")

    t_end = time.time()
    wall_time = t_end - t_start

    if verbose:
        logger.info(f"[asr] 전체 소요 시간: {wall_time:.1f}초")

    # --- 4) MRModel에 저장 ---
    # 4.1) posterior samples
    mr_model.idata = idata # Python allows dynamic attribute creation

    # 4.2) MAP point estimate 저장 (dict)
    mr_model.map_estimate = map_estimate # Python allows dynamic attribute creation

    # 4.3) wall-time
    mr_model.wall_time = wall_time # Python allows dynamic attribute creation

    print("==== asr reached the end ====")
    return map_estimate, idata