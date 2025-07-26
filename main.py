import base64
import os
import uuid

import astrbot.api.message_components as Comp
import requests
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image
from astrbot.api.star import Context, Star, register


@register("astrbot_plugin_gemini_img", "长安某", "gemini画图工具", "1.1.0")
class GeminiImageGenerator(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        """Gemini 图片生成与编辑插件初始化"""
        super().__init__(context)
        self.config = config

        logger.info(f"插件配置加载成功: {self.config}")

        # 读取多密钥配置
        self.api_keys = self.config.get("gemini_api_keys", [])
        self.current_key_index = 0

        # 初始化图片保存目录
        plugin_dir = os.path.dirname(__file__)
        self.save_dir = os.path.join(plugin_dir, "temp_images")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            logger.info(f"已创建图片临时目录: {self.save_dir}")

        # 初始化配置
        self.api_base_url = self.config.get(
            "api_base_url", "https://generativelanguage.googleapis.com"
        )

        if not self.api_keys:
            logger.error("未配置任何 Gemini API 密钥，请在插件配置中填写")

    def _get_current_api_key(self):
        """获取当前使用的 API 密钥"""
        if not self.api_keys:
            return None
        return self.api_keys[self.current_key_index]

    def _switch_next_api_key(self):
        """切换到下一个 API 密钥"""
        if not self.api_keys:
            return
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        logger.info(f"已切换到下一个 API 密钥（索引：{self.current_key_index}）")

    @filter.command("gemini_image", alias={"文生图"})
    async def generate_image(self, event: AstrMessageEvent, prompt: str):
        """根据文本描述生成图片"""
        if not self.api_keys:
            yield event.plain_result("错误：未配置任何 Gemini API 密钥")
            return

        if not prompt.strip():
            yield event.plain_result(
                "请输入图片描述（示例：/gemini_image 一只戴帽子的猫在月球上）"
            )
            return

        save_path = None

        try:
            yield event.plain_result("正在生成图片，请稍等...")
            image_data = await self._generate_image_with_retry(prompt)

            if not image_data:
                logger.error("生成失败：所有API密钥均尝试完毕")
                yield event.plain_result("生成失败：所有API密钥均尝试失败")
                return

            # 保存图片
            file_name = f"{uuid.uuid4()}.png"
            save_path = os.path.join(self.save_dir, file_name)

            with open(save_path, "wb") as f:
                f.write(image_data)

            logger.info(f"图片已保存至: {save_path}")

            # 发送图片
            yield event.chain_result([Image.fromFileSystem(save_path)])
            logger.info(f"图片发送成功，提示词: {prompt}")

        except Exception as e:
            logger.error(f"图片处理失败: {str(e)}")
            yield event.plain_result(f"生成失败: {str(e)}")

        finally:
            if save_path and os.path.exists(save_path):
                try:
                    os.remove(save_path)
                    logger.info(f"已删除临时图片: {save_path}")
                except Exception as e:
                    logger.warning(f"删除临时图片失败: {str(e)}")

    @filter.command("gemini_edit", alias={"图编辑"})
    async def edit_image(self, event: AstrMessageEvent, prompt: str):
        """仅支持：引用图片（长按回复）后发送指令编辑图片"""
        if not self.api_keys:
            yield event.plain_result("错误：未配置任何 Gemini API 密钥")
            return

        # 复用图片提取逻辑（和工具调用共用）
        image_path = await self._extract_image_from_reply(event)
        if not image_path:
            yield event.plain_result("未找到图片，请先长按图片发送回复后重试")
            return

        # 图片编辑处理
        async for result in self._process_image_edit(event, prompt, image_path):
            yield result

    @filter.llm_tool(name="edit_image")
    async def edit_image_tool(self, event: AstrMessageEvent, prompt: str):
        """编辑现有图片。使用方式：长按图片后点击“回复”，在回复框输入编辑指令（例如：把猫咪改成黑色）。

        Args:
            prompt(string): 编辑描述（例如：把猫咪改成黑色、将天空改为蓝色）
        """
        if not self.api_keys:
            yield event.plain_result("错误：未配置任何 Gemini API 密钥")
            return

        if not prompt.strip():
            yield event.plain_result("请提供编辑描述（例如：把猫咪改成黑色）")
            return

        image_path = await self._extract_image_from_reply(event)
        if not image_path:
            yield event.plain_result(
                "未找到图片，请先长按图片并点击“回复”，再输入编辑指令"
            )
            return

        async for result in self._process_image_edit(event, prompt, image_path):
            yield result

    @filter.llm_tool(name="generate_image")
    async def generate_image_tool(self, event: AstrMessageEvent, prompt: str):
        """根据文本描述生成图片。

        Args:
            prompt(string): 图片描述文本（例如：一只戴帽子的猫在月球上）
        """
        async for result in self.generate_image(event, prompt):
            yield result

    # 提取回复中图片
    async def _extract_image_from_reply(self, event: AstrMessageEvent):
        """从回复消息中提取图片并返回本地路径"""
        try:
            message_components = event.message_obj.message

            # 寻找Reply（回复）组件（用户长按图片后的“回复”操作会生成）
            reply_component = None
            for comp in message_components:
                if isinstance(comp, Comp.Reply):
                    reply_component = comp
                    logger.info(f"检测到回复消息（ID：{comp.id}），提取被引用图片")
                    break

            if not reply_component:
                logger.warning("未检测到回复组件（用户未长按图片回复）")
                return None

            # 从回复的chain中提取Image组件
            image_component = None
            for quoted_comp in reply_component.chain:
                if isinstance(quoted_comp, Comp.Image):
                    image_component = quoted_comp
                    logger.info(
                        f"从回复中提取到图片组件（file：{image_component.file}）"
                    )
                    break

            if not image_component:
                logger.warning("回复中未包含图片组件")
                return None

            # 获取本地图片路径（自动处理下载/转换）
            image_path = await image_component.convert_to_file_path()
            logger.info(f"图片已处理为本地路径：{image_path}")
            return image_path

        except Exception as e:
            logger.error(f"提取图片失败: {str(e)}", exc_info=True)
            return None

    # 统一的图片编辑处理逻辑（指令和工具调用共用）
    async def _process_image_edit(
        self, event: AstrMessageEvent, prompt: str, image_path: str
    ):
        """处理图片编辑的核心逻辑（复用给指令和工具调用）"""
        save_path = None
        try:
            yield event.plain_result("正在编辑图片，请稍等...")

            # 调用带重试的编辑方法
            image_data = await self._edit_image_with_retry(prompt, image_path)

            if not image_data:
                yield event.plain_result("编辑失败：所有API密钥均尝试失败")
                return

            # 保存并发送编辑后的图片
            save_path = os.path.join(self.save_dir, f"{uuid.uuid4()}_edited.png")
            with open(save_path, "wb") as f:
                f.write(image_data)

            yield event.chain_result([Comp.Image.fromFileSystem(save_path)])
            logger.info(f"图片编辑完成并发送，提示词: {prompt}")

        except Exception as e:
            logger.error(f"图片编辑出错：{str(e)}")
            yield event.plain_result(f"图片编辑失败：{str(e)}")

        finally:
            # 清理临时文件
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    logger.info(f"已删除原始图片临时文件：{image_path}")
                except Exception as e:
                    logger.warning(f"删除原始图片失败：{str(e)}")

            if save_path and os.path.exists(save_path):
                try:
                    os.remove(save_path)
                    logger.info(f"已删除编辑图临时文件：{save_path}")
                except Exception as e:
                    logger.warning(f"删除编辑图失败：{str(e)}")

    async def _edit_image_with_retry(self, prompt, image_path):
        """带重试逻辑的图片编辑方法"""
        max_attempts = len(self.api_keys)
        attempts = 0

        while attempts < max_attempts:
            current_key = self._get_current_api_key()
            if not current_key:
                break

            logger.info(
                f"尝试编辑图片（密钥索引：{self.current_key_index}，尝试次数：{attempts + 1}/{max_attempts}）"
            )

            try:
                return await self._edit_image_manually(prompt, image_path, current_key)
            except Exception as e:
                attempts += 1
                logger.error(f"第{attempts}次尝试失败：{str(e)}")
                if attempts < max_attempts:
                    self._switch_next_api_key()
                else:
                    logger.error("所有API密钥均尝试失败")

        return None

    async def _generate_image_with_retry(self, prompt):
        """带重试逻辑的图片生成方法"""
        max_attempts = len(self.api_keys)
        attempts = 0

        while attempts < max_attempts:
            current_key = self._get_current_api_key()
            if not current_key:
                break

            logger.info(
                f"尝试生成图片（密钥索引：{self.current_key_index}，尝试次数：{attempts + 1}/{max_attempts}）"
            )

            try:
                return await self._generate_image_manually(prompt, current_key)
            except Exception as e:
                attempts += 1
                logger.error(f"第{attempts}次尝试失败：{str(e)}")
                if attempts < max_attempts:
                    self._switch_next_api_key()
                else:
                    logger.error("所有API密钥均尝试失败")

        return None

    async def _edit_image_manually(self, prompt, image_path, api_key):
        """使用手动请求编辑图片"""
        model_name = "gemini-2.0-flash-preview-image-generation"

        # 修正API地址格式
        base_url = self.api_base_url.strip()
        if not base_url.startswith("https://"):
            base_url = f"https://{base_url}"
        if base_url.endswith("/"):
            base_url = base_url[:-1]

        endpoint = (
            f"{base_url}/v1beta/models/{model_name}:generateContent?key={api_key}"
        )
        logger.info(f"手动请求地址：{endpoint}")

        headers = {"Content-Type": "application/json"}

        # 读取图片并转换为Base64
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            image_base64 = (
                base64.b64encode(image_bytes)
                .decode("utf-8")
                .replace("\n", "")
                .replace("\r", "")
            )

        # 构建请求参数
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]},
                {
                    "role": "user",
                    "parts": [
                        {"inlineData": {"mimeType": "image/png", "data": image_base64}}
                    ],
                },
            ],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "temperature": 0.8,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 1024,
            },
        }

        # 发送请求
        response = requests.post(endpoint, headers=headers, json=payload)
        if response.status_code != 200:
            logger.error(
                f"API编辑请求失败: HTTP {response.status_code}, 响应: {response.text}"
            )
            response.raise_for_status()

        # 解析响应
        data = response.json()
        image_data = None
        if "candidates" in data and len(data["candidates"]) > 0:
            for part in data["candidates"][0]["content"]["parts"]:
                if "inlineData" in part and "data" in part["inlineData"]:
                    base64_str = (
                        part["inlineData"]["data"].replace("\n", "").replace("\r", "")
                    )
                    image_data = base64.b64decode(base64_str)
                    break

        if not image_data:
            raise Exception("编辑图片成功，但未获取到图片数据")
        return image_data

    async def _generate_image_manually(self, prompt, api_key):
        """使用手动请求生成图片"""
        model_name = "gemini-2.0-flash-preview-image-generation"

        base_url = self.api_base_url.strip()
        if not base_url.startswith("https://"):
            base_url = f"https://{base_url}"
        if base_url.endswith("/"):
            base_url = base_url[:-1]

        endpoint = (
            f"{base_url}/v1beta/models/{model_name}:generateContent?key={api_key}"
        )
        headers = {"Content-Type": "application/json"}

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "temperature": 0.8,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 1024,
            },
        }

        response = requests.post(endpoint, headers=headers, json=payload)
        if response.status_code != 200:
            logger.error(
                f"API请求失败: HTTP {response.status_code}, 响应: {response.text}"
            )
            response.raise_for_status()

        data = response.json()
        image_data = None
        if "candidates" in data and len(data["candidates"]) > 0:
            for part in data["candidates"][0]["content"]["parts"]:
                if "inlineData" in part and "data" in part["inlineData"]:
                    base64_str = (
                        part["inlineData"]["data"].replace("\n", "").replace("\r", "")
                    )
                    image_data = base64.b64decode(base64_str)
                    break

        if not image_data:
            raise Exception("生成图片成功，但未获取到图片数据")
        return image_data

    async def terminate(self):
        """插件卸载时清理临时目录"""
        if os.path.exists(self.save_dir):
            try:
                for file in os.listdir(self.save_dir):
                    os.remove(os.path.join(self.save_dir, file))
                os.rmdir(self.save_dir)
                logger.info(f"插件卸载：已清理临时目录 {self.save_dir}")
            except Exception as e:
                logger.warning(f"清理临时目录失败: {str(e)}")
        logger.info("Gemini 文生图插件已停用")
