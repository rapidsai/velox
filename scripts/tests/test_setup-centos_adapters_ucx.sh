#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

TEST_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TEST_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/velox-ucx-installer.XXXXXX")
trap 'rm -rf -- "${TEST_ROOT}"' EXIT

export DEPENDENCY_DIR="${TEST_ROOT}/dependencies"
export INSTALL_PREFIX="${TEST_ROOT}/install"
export BUILD_THREADS=1
mkdir -p "${DEPENDENCY_DIR}" "${INSTALL_PREFIX}"

# shellcheck source=/dev/null
source "${TEST_SCRIPT_DIR}/../setup-centos-adapters.sh"
set +x

populate_valid_prefix() {
  local prefix=$1
  local libdir=$2
  local package_file

  mkdir -p "${prefix}/${libdir}/ucx" "${prefix}/${libdir}/cmake/ucx"
  touch \
    "${prefix}/${libdir}/ucx/libuct_cuda.so" \
    "${prefix}/${libdir}/ucx/libuct_ib_efa.so"
  for package_file in \
    ucx-config.cmake \
    ucx-config-version.cmake \
    ucx-targets.cmake; do
    printf 'test\n' >"${prefix}/${libdir}/cmake/ucx/${package_file}"
  done
}

dnf_install() {
  return 0
}

prepare_ucx_source() {
  mkdir -p "${DEPENDENCY_DIR}/ucx/contrib"
  ln -s /bin/true "${DEPENDENCY_DIR}/ucx/contrib/configure-release"
}

wget_and_untar() {
  prepare_ucx_source
}

# Recreate the caller shape that disables Bash's implicit errexit handling
# inside functions. A failed build must remain a failure even if subsequent
# validation commands could otherwise return success.
MAKE_FAILURE_PHASE=build
make() {
  if [ "${MAKE_FAILURE_PHASE}" = build ] && [[ ${1:-} == -j* ]]; then
    return 73
  fi
  if [ "${MAKE_FAILURE_PHASE}" = build ] && [ "${1:-}" = install ]; then
    touch "${TEST_ROOT}/unexpected-install"
    return 0
  fi
  if [ "${MAKE_FAILURE_PHASE}" = install ]; then
    case ${1:-} in
      -j*)
        return 0
        ;;
      install)
        return 74
        ;;
    esac
  fi
  return 0
}

export VELOX_UCX_LOCAL_SOURCE=""
# A failed pinned-source checkout must not fall through to an old dependency
# tree, even when the caller disables Bash's implicit errexit with an AND-list.
export VELOX_UCX_VERSION=462c56777aaf268d7daf1b5d43e6f49e69b0207e
checkout_s3_direct_receive_dependency() {
  [[ $1 == kjmph/ucx && $2 == "$VELOX_UCX_VERSION" && $3 == ucx ]] || return 76
  return 75
}
masked_success=false
install_ucx && masked_success=true
status=$?
if [[ $masked_success == true || $status -ne 75 ]]; then
  echo "install_ucx did not propagate pinned-source checkout failure: ${status}" >&2
  exit 1
fi

export VELOX_UCX_VERSION=test
populate_valid_prefix "${INSTALL_PREFIX}" lib
masked_success=false
install_ucx && masked_success=true
status=$?
if [ "${masked_success}" = true ]; then
  echo "install_ucx masked a failed make command" >&2
  exit 1
fi
if [ "${status}" -ne 73 ]; then
  echo "install_ucx returned ${status}; expected the make status 73" >&2
  exit 1
fi
if [ -e "${TEST_ROOT}/unexpected-install" ]; then
  echo "install_ucx continued to make install after the failed build" >&2
  exit 1
fi

export DEPENDENCY_DIR="${TEST_ROOT}/dependencies-install"
mkdir -p "${DEPENDENCY_DIR}"
MAKE_FAILURE_PHASE=install
masked_success=false
install_ucx && masked_success=true
status=$?
if [ "${masked_success}" = true ]; then
  echo "install_ucx masked a failed make install command" >&2
  exit 1
fi
if [ "${status}" -ne 74 ]; then
  echo "install_ucx returned ${status}; expected the make install status 74" >&2
  exit 1
fi

for libdir in lib lib64; do
  valid_prefix="${TEST_ROOT}/valid-${libdir}"
  populate_valid_prefix "${valid_prefix}" "${libdir}"
  verify_local_ucx_install "${valid_prefix}"

  for package_file in \
    ucx-config.cmake \
    ucx-config-version.cmake \
    ucx-targets.cmake; do
    package_path="${valid_prefix}/${libdir}/cmake/ucx/${package_file}"
    mv "${package_path}" "${package_path}.saved"
    if verify_local_ucx_install "${valid_prefix}" 2>"${TEST_ROOT}/verify.err"; then
      echo "UCX validation accepted missing metadata: ${package_path}" >&2
      exit 1
    fi
    grep -F "${package_path}" "${TEST_ROOT}/verify.err" >/dev/null
    mv "${package_path}.saved" "${package_path}"
  done
done

echo "test_setup-centos_adapters_ucx.sh: PASS"
