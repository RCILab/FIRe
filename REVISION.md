# `revision-ral` branch 안내 (공저자용)

이 GitHub branch에는 revision 작업 중 바뀐 **코드만** 올린다. `RCILab/FIRe`는 공개 repo라서, 리뷰 대응 자료(작업 계획, 코멘트별 대응, 결과 note)와 실험 데이터는 여기 올리지 않고 비공개 Google Drive 폴더로 공유한다.

## Drive 폴더 `revision-logs`
- 링크: https://drive.google.com/drive/folders/11mwesH6AHgN-XjwOnu93FOh5tqNxh455
- 이메일로 초대받은 계정만 열 수 있다. 접근 권한은 @chohh7391에게 요청한다.
- 구성:

| 위치 | 내용 |
|---|---|
| `README.md` | 폴더 구조와 사용법. **먼저 읽는다** |
| `docs/` | 작업 계획(`PLAN.md`), 코멘트별 대응 현황, task별 결과 note, 작업 규칙(`CLAUDE.md`)을 git 없이 읽을 수 있는 사본 |
| `analysis/` | 분석 결과(그림, 통계)와 스크립트 |
| `git/FIRe-revision-ral.bundle` | 코드와 revision 자료가 모두 들어 있는 `revision-ral` 전체. 이 GitHub branch보다 내용이 많다 |
| `results/` | 실험 로그, checkpoint, 데이터셋 (약 55 GB) |

## 작업 시작
1. Drive에서 `git/FIRe-revision-ral.bundle`을 받는다.
2. bundle의 branch로 개인 branch를 만든다.
   ```bash
   # 새로 받을 때
   git clone -b revision-ral /path/to/FIRe-revision-ral.bundle FIRe
   cd FIRe && git checkout -b revision-ral-<id>        # <id>: bhe1004 또는 chohh7391

   # 이 GitHub repo를 이미 clone했을 때
   git fetch /path/to/FIRe-revision-ral.bundle revision-ral
   git checkout -b revision-ral-<id> FETCH_HEAD
   ```
3. 그 branch의 `CLAUDE.md`(작업 규칙, branch 규칙, Drive 갱신 방법)와 `PLAN.md`(할 일 목록)를 읽는다. Claude Code는 `CLAUDE.md`를 자동으로 읽는다.
4. 큰 데이터는 rclone으로 받는다. 방법은 Drive의 `README.md`에 있다. OAuth client 정보(client_id, secret)는 repo와 Drive에 두지 않으므로 @chohh7391에게 직접 받는다.
5. 작업한 개인 branch는 bundle로 만들어 Drive `git/`에 올리고, 상의한 뒤 `revision-ral`에 반영한다.

## 공개 repo 주의
- `revision/`, `PLAN.md`, revision용 `CLAUDE.md`, `results/`가 들어 있는 branch는 이 repo에 push하지 않는다.
- 코드만 올릴 때는 bundle 안 `CLAUDE.md`의 "push 절차"를 따른다.
- 이 파일과 `README.md` 맨 위의 안내 줄은 revision 작업용이다. `main`에 반영할 때는 뺀다.
