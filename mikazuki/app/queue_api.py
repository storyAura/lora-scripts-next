"""REST surface for the sequential training queue (mounted under /api/queue)."""

import json

from fastapi import APIRouter, Request

from mikazuki.app.models import APIResponseFail, APIResponseSuccess
from mikazuki.train_queue import train_queue

router = APIRouter(prefix="/queue")


async def _json_body(request: Request) -> dict:
    try:
        payload = json.loads(await request.body() or b"{}")
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


@router.get("")
async def queue_state():
    return APIResponseSuccess(message=None, data=train_queue.snapshot())


@router.post("/start")
async def queue_start(request: Request):
    payload = await _json_body(request)
    return train_queue.start(include_paused=bool(payload.get("include_paused")))


@router.post("/stop")
async def queue_stop():
    return train_queue.stop()


@router.post("/reorder")
async def queue_reorder(request: Request):
    payload = await _json_body(request)
    ids = payload.get("ids")
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        return APIResponseFail(message="ids 必须是任务 id 列表")
    return train_queue.reorder(ids)


@router.post("/clear-finished")
async def queue_clear_finished():
    return train_queue.clear_finished()


@router.get("/entries/{entry_id}/config")
async def queue_entry_config(entry_id: str):
    config = train_queue.entry_config(entry_id)
    if config is None:
        return APIResponseFail(message="任务不存在或已被移除")
    return APIResponseSuccess(message=None, data={"config": config})


@router.post("/entries/{entry_id}/pause")
async def queue_entry_pause(entry_id: str):
    return train_queue.pause(entry_id)


@router.post("/entries/{entry_id}/resume")
async def queue_entry_resume(entry_id: str):
    return train_queue.resume(entry_id)


@router.post("/entries/{entry_id}/requeue")
async def queue_entry_requeue(entry_id: str):
    return train_queue.requeue(entry_id)


@router.post("/entries/{entry_id}/start")
async def queue_entry_start(entry_id: str):
    return train_queue.start_entry(entry_id)


@router.post("/entries/{entry_id}/editing")
async def queue_entry_editing(entry_id: str, request: Request):
    payload = await _json_body(request)
    return train_queue.set_editing(entry_id, bool(payload.get("editing")))


@router.delete("/entries/{entry_id}")
async def queue_entry_delete(entry_id: str):
    return train_queue.remove(entry_id)
