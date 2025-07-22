import logging
from instaloader import instaloader, InstaloaderException

from app.exceptions import InstagramError

logger = logging.getLogger(__name__)


class InstagramComponent:
    def __init__(self, username: str):
        self.L = instaloader.Instaloader()
        self.username = username

    def login(self):
        logger.info(f"[{self.username}] 인스타그램 세션 파일로 로그인 시도")
        try:
            self.L.load_session_from_file(self.username)
            logger.info(f"[{self.username}] 인스타그램 세션 파일로 성공적으로 로그인했습니다.")
        except FileNotFoundError:
            logger.warning(
                f"세션 파일을 찾을 수 없습니다: ~/.config/instaloader/session-{self.username}"
            )
            logger.warning("로그인하지 않고 진행합니다. 공개 프로필의 데이터만 접근 가능합니다.")
        except InstaloaderException as e:
            raise InstagramError(f"세션 파일 로딩 중 오류 발생: {e}") from e