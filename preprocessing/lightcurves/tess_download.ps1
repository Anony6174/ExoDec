$files = Get-Content "all_fits.txt"
$downloadFolder = "all_lcs"
if (-not (Test-Path $downloadFolder)) {
    New-Item -ItemType Directory -Path $downloadFolder
}
# Download each file
Set-Location $downloadFolder


foreach ($file in $files) {
    $url = "https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:TESS/product/$file"
    Write-Host "Downloading $file..."
    curl.exe -C - -L -o $file $url
}

# $files = @(
#     # "tess2018206045859-s0001-0000000278660115-0120-s_lc.fits",
#     # "tess2018206045859-s0001-0000000114952667-0120-s_lc.fits",
#     # "tess2018206045859-s0001-0000000053841444-0120-s_lc.fits",
#     # "tess2018206045859-s0001-0000000140659172-0120-s_lc.fits",
#     # "tess2018206045859-s0001-0000000294053203-0120-s_lc.fits",
#     # "tess2018206045859-s0001-0000000197764361-0120-s_lc.fits",
#     # "tess2018206045859-s0001-0000000238202274-0120-s_lc.fits",
#     # "tess2018206045859-s0001-0000000062483415-0120-s_lc.fits",
#     # "tess2018206045859-s0001-0000000355369707-0120-s_lc.fits",
#     # "tess2018206045859-s0001-0000000234516451-0120-s_lc.fits",
#     # "tess2018206045859-s0001-0000000055745413-0120-s_lc.fits",
#     # "tess2018206045859-s0001-0000000302116223-0120-s_lc.fits",
#     # "tess2018206045859-s0001-0000000038600576-0120-s_lc.fits",
#     # "tess2018206045859-s0001-0000000382043075-0120-s_lc.fits",
#     # "tess2018206045859-s0001-0000000214568827-0120-s_lc.fits",
#     "tess2018206045859-s0001-0000000038510224-0120-s_lc.fits"
# )
# #save output in sec1 folder
# $downloadFolder = "sec1"
# if (-not (Test-Path $downloadFolder)) {
#     New-Item -ItemType Directory -Path $downloadFolder
# }
# # Download each file
# Set-Location $downloadFolder

# foreach ($file in $files) {
#     $url = "https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:TESS/product/$file"
#     Write-Host "Downloading $file..."
#     curl.exe -C - -L -o $file $url
# }

