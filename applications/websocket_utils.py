# applications/websocket_utils.py
from fastapi import WebSocket
from typing import Dict, List, Optional
from config import logger

class ConnectionManager:
    def __init__(self):
        # user_id -> WebSocket connection
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info(f"WebSocket connected for user_id: {user_id}")

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            logger.info(f"WebSocket disconnected for user_id: {user_id}")

    async def send_personal_message(self, message: dict, user_id: str):
        websocket = self.active_connections.get(user_id)
        if websocket:
            try:
                await websocket.send_json(message)
                # logger.debug(f"Sent WebSocket message to {user_id}: {message}")
            except Exception as e:
                logger.error(f"Error sending WebSocket message to {user_id}: {e}. Connection might be closed.")
                # Optionally try to disconnect if send fails repeatedly
                # self.disconnect(user_id) 
        else:
            logger.warning(f"No active WebSocket connection found for user_id: {user_id} to send message: {message}")

manager = ConnectionManager()
