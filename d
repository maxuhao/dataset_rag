[1mdiff --git a/app/import_process/agent/nodes/node_pdf_to_md.py b/app/import_process/agent/nodes/node_pdf_to_md.py[m
[1mindex dd7a7eb..6bf957c 100644[m
[1m--- a/app/import_process/agent/nodes/node_pdf_to_md.py[m
[1m+++ b/app/import_process/agent/nodes/node_pdf_to_md.py[m
[36m@@ -1,8 +1,13 @@[m
[32m+[m[32mimport shutil[m
 import sys[m
[32m+[m[32mimport time[m
[32m+[m[32mimport zipfile[m
 from pathlib import Path[m
 [m
[32m+[m[32mimport requests[m
 from sqlalchemy.testing.util import function_named[m
 [m
[32m+[m[32mfrom app.conf.mineru_config import mineru_config[m
 from app.core.logger import logger[m
 from app.import_process.agent.state import ImportGraphState[m
 from app.utils.task_utils import add_running_task, add_done_task[m
[36m@@ -41,12 +46,174 @@[m [mdef step_1_validate_paths(state):[m
     return pdf_path_obj, local_dir_obj[m
 [m
 [m
[31m-def step_2_upload_and_poll(pdf_path_obj):[m
[31m-    pass[m
[32m+[m[32mdef step_2_upload_and_poll(pdf_path_obj) -> str:[m
[32m+[m[32m    """[m
[32m+[m[32m    将pdf文件使用minerU解析，并且获取md对应的下载的url地址！！[m
[32m+[m[32m    :param pdf_path_obj: 上传解析pdf文件的 path对象[m
[32m+[m[32m    :return: str -> url , minerU解析后md文件zip压缩包的下载地址[m
[32m+[m[32m    """[m
[32m+[m[32m    token = mineru_config.api_key[m
[32m+[m[32m    url = f"{mineru_config.base_url}/file-urls/batch"[m
[32m+[m[32m    header = {[m
[32m+[m[32m        "Content-Type": "application/json",[m
[32m+[m[32m        "Authorization": f"Bearer {token}"[m
[32m+[m[32m    }[m
[32m+[m[32m    data = {[m
[32m+[m[32m        "files": [[m
[32m+[m[32m            {"name": f"{pdf_path_obj.name}"}[m
[32m+[m[32m        ],[m
[32m+[m[32m        "model_version": "vlm"[m
[32m+[m[32m    }[m
[32m+[m[32m    response = requests.post(url, headers=header, json=data)[m
[32m+[m[32m    # 结果处理 请求http状态码不是200 或者 返回结果的状态码 不是0 请求失败！[m
[32m+[m[32m    if response.status_code != 200 or response.json()['code'] != 0:[m
[32m+[m[32m        logger.error(f"[step_2_upload_and_poll]请求minerU解析接口失败，请检查输入文件路径是否正确！！")[m
[32m+[m[32m        raise RuntimeError(f"[step_2_upload_and_poll]请求minerU解析接口失败，请检查输入文件路径是否正确！！")[m
[32m+[m[32m    uploaded_url = response.json()['data']['file_urls'][0] # 上这个地址上传文件[m
[32m+[m[32m    batch_id = response.json()['data']['batch_id'] # 处理id，后续根据这个id获取结果！[m
[32m+[m
[32m+[m[32m    # 2. 将文件上传到对应的解析地址[m
[32m+[m[32m    # 使用Put请求，将pdf_path_obj文件传递到uploaded_url地址即可！[m
[32m+[m[32m    # 注意： 不能直接使用put! 这块很大概率报错！ 原因：电脑开了各种代理，put的请求头，添加一些额外的参数头！将文件真的转存到第三方的文件存储服务器！[m
[32m+[m[32m    # 文件存储服务器检查都比较严格！ 拒绝存储！报错！ get post 宽进宽出  put严进严出！[m
[32m+[m[32m    http_session = requests.Session()[m
[32m+[m[32m    http_session.trust_env = False # 禁用代理 复用请求对象[m
[32m+[m[32m    try:[m
[32m+[m[32m        with open(pdf_path_obj, 'rb') as f:[m
[32m+[m[32m            file_data = f.read()[m
[32m+[m[32m        upload_response = http_session.put(uploaded_url, data=file_data)[m
[32m+[m[32m        if upload_response.status_code != 200:[m
[32m+[m[32m            logger.error(f"[step_2_upload_and_poll]上传文件失败，请检查输入文件路径是否正确！！")[m
[32m+[m[32m            raise RuntimeError(f"[step_2_upload_and_poll]上传文件失败，请检查输入文件路径是否正确！！")[m
[32m+[m[32m    except Exception as e:[m
[32m+[m[32m        logger.error(f"[step_2_upload_and_poll]上传文件失败，请检查输入文件路径是否正确！！")[m
[32m+[m[32m        raise RuntimeError(f"[step_2_upload_and_poll]上传文件失败，请检查输入文件路径是否正确！！")[m
[32m+[m[32m    finally:[m
[32m+[m[32m        http_session.close()[m
[32m+[m[32m    # 3. 轮询获取解析结果[m
[32m+[m[32m    # 循环获取！确保获取到结果，再先后执行！！[m
[32m+[m[32m    # 设计一个循环，3秒获取一次！ 最多等待10分钟600 -> 600页pdf[m
[32m+[m
[32m+[m[32m    url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"[m
[32m+[m[32m    timeout_seconds = 600  # 1s -> 1页pdf[m
[32m+[m[32m    poll_interval = 3  #间隔时间是3秒[m
[32m+[m[32m    start_time = time.time() # 进去起始时间[m
[32m+[m
[32m+[m[32m    while True:[m
[32m+[m[32m        # 3.1 超时判断 不能站在第一次角度！ 站在宏观的角度！[m
[32m+[m[32m        if time.time() - start_time > timeout_seconds:[m
[32m+[m[32m            logger.error(f"[step_2_upload_and_poll]请求minerU解析接口超时，请检查输入文件路径是否正确！！")[m
[32m+[m[32m            raise RuntimeError(f"[step_2_upload_and_poll]请求minerU解析接口超时，请检查输入文件路径是否正确！！")[m
[32m+[m[32m        # 3.2 向指定url地址获取本次解析的结果[m
[32m+[m[32m        res = requests.get(url, headers=header)[m
[32m+[m[32m        # 3.3 解析结果判断和获取zip_url[m
[32m+[m[32m        if res.status_code != 200:[m
[32m+[m[32m            # 5xx系列我们会给与机会，直到timeout[m
[32m+[m[32m            # http的代码 1xx 2xx 3xx 4xx 【5xx】[m
[32m+[m[32m            if 500 <= res.status_code < 600:[m
[32m+[m[32m                # 5xx系列我们会给与机会，直到timeout[m
[32m+[m[32m                time.sleep(poll_interval)[m
[32m+[m[32m                continue[m
[32m+[m[32m            raise RuntimeError(f"[step_2_upload_and_poll]请求minerU解析接口失败，返回的状态码{res.status_code}！！")[m
[32m+[m
[32m+[m[32m        json_data = res.json()[m
[32m+[m[32m        if json_data['code'] != 0:[m
[32m+[m[32m            # != 0 很大概token过期了，后续没有钱了[m
[32m+[m[32m            raise RuntimeError([m
[32m+[m[32m                f"[step_2_upload_and_poll]请求minerU解析接口失败，返回的错误:{json_data['code']}信息{json_data['msg']}！！")[m
[32m+[m
[32m+[m[32m        # 判断下解析状态[m
[32m+[m[32m        extract_result = json_data['data']['extract_result'][0][m
[32m+[m[32m        if extract_result['state'] == 'done':[m
[32m+[m[32m            # 解析完毕后获取结果[m
[32m+[m[32m            full_zip_url = extract_result['full_zip_url'][m
[32m+[m[32m            logger.info(f"已经完成pdf的解析，耗时：{time.time()-start_time}s,解析结果：{full_zip_url}")[m
[32m+[m[32m            return full_zip_url[m
[32m+[m[32m        else:[m
[32m+[m[32m            # 还没解析完成[m
[32m+[m[32m            time.sleep(poll_interval)[m
 [m
 [m
 def step_3_download_and_extract(zip_url, local_dir_obj, stem):[m
[31m-    pass[m
[32m+[m[32m    """[m
[32m+[m[32m    下载指定的md.zip文件，并且解压，返回解压后的md文件的地址！[m
[32m+[m[32m    :param zip_url:  要下载的地址[m
[32m+[m[32m    :param local_dir_obj: 存储的文件夹[m
[32m+[m[32m    :param stem: pdf的文件名字[m
[32m+[m[32m    :return: 返回md文件的地址[m
[32m+[m[32m    """[m
[32m+[m[32m    # 1. 下载zip文件 response响应体[m
[32m+[m[32m    response = requests.get(zip_url)[m
[32m+[m
[32m+[m[32m    if response.status_code != 200:[m
[32m+[m[32m        logger.error(f"[step_3_download_and_extract]下载文件失败，请检查输入文件路径是否正确！！")[m
[32m+[m[32m        raise RuntimeError(f"[step_3_download_and_extract]下载文件失败，请检查输入文件路径是否正确！！")[m
[32m+[m[32m    # 2. 将响应体的zip文件保存到本地[m
[32m+[m[32m    # 保存文件 output/二狗子/ 二狗子_result.zip[m
[32m+[m[32m    zip_save_path = local_dir_obj / f"{stem}_result.zip"[m
[32m+[m[32m    with open(zip_save_path, 'wb') as f:[m
[32m+[m[32m        f.write(response.content)[m
[32m+[m[32m    logger.info(f"[step_3_download_and_extract]下载文件成功，保存路径：{zip_save_path}")[m
[32m+[m
[32m+[m[32m    # 3. 清空下旧目录（将上一次处理的文件目录进行删除）[m
[32m+[m[32m    extract_target_dir = local_dir_obj / stem[m
[32m+[m
[32m+[m[32m    # 先清空旧目录因为两次解压的文件数量可能不一样，会保留旧数据[m
[32m+[m[32m    if extract_target_dir.exists():[m
[32m+[m[32m        # 通过递归进行删除 本身也会被删除[m
[32m+[m[32m        shutil.rmtree(extract_target_dir)[m
[32m+[m[32m    # 创建一个新的目录[m
[32m+[m[32m    extract_target_dir.mkdir(parents=True, exist_ok=True) # parents=True 创建父目录 exist_ok=True 存在则不创建[m
[32m+[m
[32m+[m[32m    # 4. 解压zip文件[m
[32m+[m[32m    # python zip解压模块 zipfile进行zip压缩和解压[m
[32m+[m[32m    # zipfile 处理zip文件的模块[m
[32m+[m[32m    # zip_file_object 创建一个zip文件对象 只能读取 解压[m
[32m+[m[32m    with zipfile.ZipFile(zip_save_path, 'r') as zip_file_object:[m
[32m+[m[32m        # 调用对象的解压方法进行解压即可 参数：解压的目标文件夹 output/二狗子[m
[32m+[m[32m        zip_file_object.extractall(extract_target_dir)[m
[32m+[m[32m    # 5 返回md文件的地址[m
[32m+[m[32m    # 解压后得到的文件明 可能叫 文化.md or full.md[m
[32m+[m[32m    md_file_list = list(extract_target_dir.rglob("*.md"))[m
[32m+[m
[32m+[m[32m    if not md_file_list:[m
[32m+[m[32m        logger.error(f"[step_3_download_and_extract]没有找到md文件，请检查输入文件路径是否正确！！")[m
[32m+[m[32m        raise RuntimeError(f"[step_3_download_and_extract]没有找到md文件，请检查输入文件路径是否正确！！")[m
[32m+[m
[32m+[m[32m    target_md_file = None # 最终的md文件[m
[32m+[m
[32m+[m[32m    # 检查有没有源文件明的md[m
[32m+[m[32m    for md_file in md_file_list:[m
[32m+[m[32m        # stem 文件名 二狗子[m
[32m+[m[32m        if md_file.name == stem + ".md":[m
[32m+[m[32m            target_md_file = md_file[m
[32m+[m[32m            break[m
[32m+[m[32m    # 检查有没有full.md（第一次没有找到，才找到full）[m
[32m+[m[32m    if not target_md_file:[m
[32m+[m[32m        for md_file in md_file_list:[m
[32m+[m[32m            if md_file.name.lower() == "full.md":[m
[32m+[m[32m                target_md_file = md_file[m
[32m+[m[32m                break[m
[32m+[m[32m    # 都没有 默认获取第一个即可[m
[32m+[m[32m    if not target_md_file:[m
[32m+[m[32m        target_md_file = md_file_list[0][m
[32m+[m
[32m+[m[32m    # md文件名 二狗子.md full.md 不知道.md[m
[32m+[m[32m    # 统一改成 源文件名（stem）.md[m
[32m+[m[32m    # 不是原名字的时候 才重命名[m
[32m+[m[32m    if target_md_file.name != stem:[m
[32m+[m[32m        # 进行重命名[m
[32m+[m[32m        # target_md_file.with_name(f"{stem}.md") 修改path对象 （不涉及文件操作） 返回结果是修改后path对象[m
[32m+[m[32m        # target_md_file.rename(target_md_file.with_name(f"{stem}.md")) 修改磁盘中的文件名称（修改名称了） return 新的路径path[m
[32m+[m[32m        # rename是磁盘操作，会返回新的路径path with_name是修改文件名[m
[32m+[m[32m        target_md_file = target_md_file.rename(target_md_file.with_name(f"{stem}.md"))[m
[32m+[m
[32m+[m[32m    # 最终的md文件获取绝对路径，并返回字符串类型[m
[32m+[m[32m    final_md_str_path = str(target_md_file.resolve())[m
[32m+[m[32m    logger.info(f"[step_3_download_and_extract]解压文件成功，保存路径：{final_md_str_path}")[m
[32m+[m[32m    return final_md_str_path[m
[32m+[m
[32m+[m
 [m
 [m
 def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:[m
