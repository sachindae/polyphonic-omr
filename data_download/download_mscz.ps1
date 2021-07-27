Import-Csv [csv_file] |
    ForEach-Object -Parallel {
        <wget path> "https://ipfs.infura.io$($_.ref)" -O "$($_.id).mscz"
    } -ThrottleLimit 16


echo done