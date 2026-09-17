def __getattr__(name: str):
    if name in ("VITA_BOY_ACTION_SCALE", "get_vita_boy_robot_cfg"):
        from .vita_boy import vita_boy_constants

        globals()[name] = getattr(vita_boy_constants, name)
        return globals()[name]
    raise AttributeError(name)
