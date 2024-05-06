import streamlit as st
from openxlab.dataset import info, query, create_repo, upload_file, upload_folder, get, download, commit, remove_repo

st.title("数据集信息预览工具")

st.sidebar.title("功能选择")
function = st.sidebar.selectbox("请选择功能", ("info", "query", "create_repo", "upload_file", "upload_folder", "get", "download", "commit", "remove_repo"))

if function == "info":
    st.header("数据集元信息查看")
    dataset_repo = st.text_input("请输入数据集仓库地址（username/repo_name）：")
    if st.button("查看"):
        result = info(dataset_repo=dataset_repo)
        st.write(result)

elif function == "query":
    st.header("数据集文件列表查看")
    dataset_repo = st.text_input("请输入数据集仓库地址（username/repo_name）：")
    if st.button("查看"):
        result = query(dataset_repo=dataset_repo)
        st.write(result)

elif function == "create_repo":
    st.header("数据集创建")
    repo_name = st.text_input("请输入数据集仓库名称：")
    if st.button("创建"):
        result = create_repo(repo_name=repo_name)
        st.write(result)

elif function == "upload_file":
    st.header("数据集上传文件")
    dataset_repo = st.text_input("请输入数据集仓库地址（username/repo_name）：")
    source_path = st.text_input("请输入上传本地文件的所在的路径：")
    target_path = st.text_input("请输入对应数据集仓库下的相对路径（可选）：")
    if st.button("上传"):
        result = upload_file(dataset_repo=dataset_repo, source_path=source_path, target_path=target_path)
        st.write(result)

elif function == "upload_folder":
    st.header("数据集上传文件夹")
    dataset_repo = st.text_input("请输入数据集仓库地址（username/repo_name）：")
    source_path = st.text_input("请输入上传本地文件夹的所在的路径：")
    target_path = st.text_input("请输入对应数据集仓库下的相对路径（可选）：")
    if st.button("上传"):
        result = upload_folder(dataset_repo=dataset_repo, source_path=source_path, target_path=target_path)
        st.write(result)

elif function == "get":
    st.header("数据集仓库下载")
    dataset_repo = st.text_input("请输入数据集仓库地址（username/repo_name）：")
    target_path = st.text_input("请输入下载仓库指定的本地路径：")
    if st.button("下载"):
        result = get(dataset_repo=dataset_repo, target_path=target_path)
        st.write(result)

elif function == "download":
    st.header("数据集文件下载")
    dataset_repo = st.text_input("请输入数据集仓库地址（username/repo_name）：")
    source_path = st.text_input("请输入对应数据集仓库下文件的相对路径：")
    target_path = st.text_input("请输入下载仓库指定的本地路径：")
    if st.button("下载"):
        result = download(dataset_repo=dataset_repo, source_path=source_path, target_path=target_path)
        st.write(result)

elif function == "commit":
    st.header("数据集提交修改")
    dataset_repo = st.text_input("请输入数据集仓库地址（username/repo_name）：")
    commit_message = st.text_input("请输入提交信息：")
    if st.button("提交"):
        result = commit(dataset_repo=dataset_repo, commit_message=commit_message)
        st.write(result)

elif function == "remove_repo":
    st.header("数据集仓库删除")
    dataset_repo = st.text_input("请输入数据集仓库地址（username/repo_name）：")
    if st.button("删除"):
        result = remove_repo(dataset_repo=dataset_repo)
        st.write(result)