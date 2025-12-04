from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
import os, sys

class MCPClient:
    def __init__(self, python="python", server_script="./mcp_server_new.py", server_args="--transport stdio"):
        self.python=python
        self.server_script = server_script
        self.server_args = server_args
        self._client_context = None
        self._session_context = None

    async def connect(self):
        cmd = [self.server_script] + self.server_args.split()
        print("Launching server with:", cmd)  # debug
        server_params = StdioServerParameters(command=self.python, args=cmd, env=None)
        self._client_ession = None
        self._read = None
        self._write = None
        self._clcontext = stdio_client(server_params)
        self._read, self._write = await self._client_context.__aenter__()

        self._session_context = ClientSession(self._read, self._write)
        self._session = await self._session_context.__aenter__()
        await self._session.initialize()

        return self._session
    
    async def disconnect(self):
        if self._session_context:
            await self._session_context.__aexit__(None,None,None)
        if self._client_context:
            await self._client_context.__aexit__(None, None, None)
    
    async def __aenter__(self):
        return await self.connect()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()