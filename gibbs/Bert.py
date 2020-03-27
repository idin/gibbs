from pdf import PDF
from linguistics.bert import BertVectorizer
from pandas import DataFrame
from pandas import concat
from chronometry.progress import iterate
from disk import Path
from ravenclaw.wrangling import bring_to_front
from joblib import Parallel, delayed


class Bert:
	def __init__(self, num_tokens=50):
		self._bert_vectorizer = BertVectorizer(num_tokens=num_tokens)

	def vectorize(self, path, ignore_if_html_file_exists=True, ignore_errors=False, num_threads=1):
		"""
		:type path: str or Path
		:type ignore_if_html_file_exists: bool
		:type ignore_errors: bool
		:rtype: DataFrame
		"""
		path = Path(path)
		pdf_paths = [
			PDF(file_path)
			for file_path in path.list(show_size=False)
			if file_path.extension.lower() == 'pdf'
		]

		exceptions = []

		# create htmls
		for pdf_path in iterate(iterable=pdf_paths, text='converting pdfs to html'):
			try:
				pdf_path.convert_to_html(ignore_if_exists=ignore_if_html_file_exists)
			except Exception as e:
				if ignore_errors:
					exceptions.append(e)
				else:
					raise e

		# get paragraphs
		def extract_paragraphs(pdf_path):
			try:
				pdf_paragraphs = pdf_path.paragraphs
				num_paragraphs = len(pdf_paragraphs)
				return [
					{'pdf': pdf_path, 'paragraph_num': i + 1, 'paragraph': paragraph, 'num_paragraphs': num_paragraphs}
					for i, paragraph in enumerate(pdf_paragraphs)
				]

			except Exception as e:
				if ignore_errors:
					return [{'error': e}]
				else:
					raise e

		if num_threads == 1:
			paragraph_dict_lists = [
				extract_paragraphs(x)
				for x in iterate(pdf_paths, text='extracting paragraphs (single-threaded)')
			]
		else:
			processor = Parallel(n_jobs=num_threads, backend='threading', require='sharedmem')
			paragraph_dict_lists = processor(
				delayed(extract_paragraphs)(pdf_path=x)
				for x in iterate(pdf_paths, text='extracting paragraphs (multi-threaded)')
			)
		paragraph_dicts = [x for paragraph_dict_list in paragraph_dict_lists for x in paragraph_dict_list]

		# create vectors
		def get_vector_and_num_tokens(paragraph_dict):
			try:
				pdf_path = paragraph_dict['pdf']
				paragraph_num = paragraph_dict['paragraph_num']
				paragraph = paragraph_dict['paragraph']
				num_paragraphs = paragraph_dict['num_paragraphs']
				vector, num_tokens = self._bert_vectorizer.vectorize(text=paragraph, get_num_tokens=True)
				vector_df = DataFrame(
					vector,
					columns=[f'bert_{i + 1}' for i in range(vector.shape[1])]
				)
				vector_df['pdf'] = pdf_path.name_and_extension
				vector_df['num_paragraphs'] = num_paragraphs
				vector_df['paragraph_num'] = paragraph_num
				vector_df['num_tokens'] = num_tokens
				return vector_df
			except Exception as e:
				if ignore_errors:
					return e
				else:
					raise e

		if num_threads == 1:
			vectors = [
				get_vector_and_num_tokens(paragraph_dict=x)
				for x in iterate(paragraph_dicts, text='converting paragraphs to vectors (single-threaded)')
			]
		else:
			processor = Parallel(n_jobs=num_threads, backend='threading', require='sharedmem')
			vectors = processor(
				delayed(get_vector_and_num_tokens)(paragraph_dict=x)
				for x in iterate(paragraph_dicts, text='converting paragraphs to vectors (multi-threaded)')
			)

		return bring_to_front(
			data=concat(vectors),
			columns=['pdf', 'paragraph_num', 'num_paragraphs', 'num_tokens']
		).reset_index(drop=True)
