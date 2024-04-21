import os
import pandas as pd

import asyncio
import aiohttp

import aiofiles


HOME = os.getcwd()
INSTR_DIR = os.path.join(HOME, 'instructions')
rceth_csv = os.path.join(HOME, 'rceth.csv')


async def download_pdf(session, pdf_url, save_dir):
    instr_name = os.path.basename(pdf_url)
    pdf_path = os.path.join(save_dir, instr_name)

    if not os.path.exists(pdf_path):
        try:
            async with session.get(pdf_url) as res:
                if res.status == 200:
                    async with aiofiles.open(pdf_path, mode='wb') as f:
                        await f.write(await res.read())
                    # print(f"Instruction {instr_name} downloaded.")
                else:
                    print(
                        f"Failed to download {instr_name}: Status {res.status}")

        except asyncio.TimeoutError:
            print("The request timed out.")


async def main():
    df = pd.read_csv(rceth_csv, encoding='windows-1251')
    links = df['link_of_instruction'].dropna().str.split(
        ',').explode().str.strip().tolist()

    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for link in links:
            task = asyncio.create_task(download_pdf(session, link, INSTR_DIR))
            tasks.append(task)
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
