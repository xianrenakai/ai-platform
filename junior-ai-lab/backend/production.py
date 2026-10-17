import os
from pathlib import Path

from starlette.exceptions import HTTPException
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

os.environ.setdefault("APP_ENV", "production")

from app.main import app


ROOT = Path(__file__).resolve().parents[1]
FRONTENDS = {
    "/recognition": ROOT / "module-recognition" / "dist-production",
    "/synthesis": ROOT / "module-synthesis" / "dist-production",
    "/cnn": ROOT.parent / "net_web" / "dist",
}
PORTAL_DIST = ROOT / "portal" / "dist-production"


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        try:
            response = await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            response = None
        if (response is None or response.status_code == 404) and (self.directory is not None):
            index = Path(self.directory) / "index.html"
            if index.exists():
                return FileResponse(index)
        if response is None:
            raise HTTPException(status_code=404)
        return response


missing = [str(path) for path in [*FRONTENDS.values(), PORTAL_DIST] if not path.exists()]
if missing:
    raise RuntimeError("前端尚未构建，请先运行 BuildAIPlatform.bat。缺少：" + ", ".join(missing))

for route, directory in FRONTENDS.items():
    app.mount(route, SPAStaticFiles(directory=str(directory), html=True), name=route.strip("/"))


async def disabled_api_documentation():
    raise HTTPException(status_code=404)


for documentation_path in ("/docs", "/redoc", "/openapi.json"):
    app.add_api_route(documentation_path, disabled_api_documentation, include_in_schema=False)

# Must be mounted last: it owns the portal root while existing /api routes keep priority.
app.mount("/", SPAStaticFiles(directory=str(PORTAL_DIST), html=True), name="portal")
