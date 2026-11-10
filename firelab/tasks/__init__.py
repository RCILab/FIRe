import gymnasium as gym

_ENV = "firelab.tasks.fire.fire_env:FIReEnv"
_CFG = "firelab.tasks.fire.fire_env_cfg"
_AGENTS = "firelab.tasks.fire.agents"

for _task, _key in (("PegInsert", "peg_insert"), ("GearMesh", "gear_mesh"), ("NutThread", "nut_thread")):
    for _id, _entry, _cfg, _agent in (
        (f"FIRe-{_task}-v0", _ENV, f"FIRe{_task}Cfg", f"rl_games_ppo_cfg_{_key}.yaml"),
        (f"FIRe-{_task}-VLAOnly-v0", _ENV, f"FIRe{_task}VLAOnlyCfg", f"rl_games_ppo_cfg_{_key}.yaml"),
    ):
        gym.register(
            id=_id,
            entry_point=_entry,
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": f"{_CFG}:{_cfg}", "rl_games_cfg_entry_point": f"{_AGENTS}:{_agent}"},
        )
